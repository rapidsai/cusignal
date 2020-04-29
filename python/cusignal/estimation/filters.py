# Copyright (c) 2019-2020, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# 1
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cupy as cp
import itertools
import numpy as np

from enum import Enum
from numba import cuda, void, float32, float64


class GPUKernel(Enum):
    PREDICT = 0
    UPDATE = 1


class GPUBackend(Enum):
    CUPY = 0
    NUMBA = 1


# Numba type supported and corresponding C type
_SUPPORTED_TYPES = {
    np.float32: [float32, "double"],
    np.float64: [float64, "double"],
}


_numba_kernel_cache = {}
_cupy_kernel_cache = {}


# Use until functionality provided in Numba 0.49/0.50 available
def stream_cupy_to_numba(cp_stream):
    """
    Notes:
        1. The lifetime of the returned Numba stream should be as
           long as the CuPy one, which handles the deallocation
           of the underlying CUDA stream.
        2. The returned Numba stream is assumed to live in the same
           CUDA context as the CuPy one.
        3. The implementation here closely follows that of
           cuda.stream() in Numba.
    """
    from ctypes import c_void_p
    import weakref

    # get the pointer to actual CUDA stream
    raw_str = cp_stream.ptr

    # gather necessary ingredients
    ctx = cuda.devices.get_context()
    handle = c_void_p(raw_str)

    # create a Numba stream
    nb_stream = cuda.cudadrv.driver.Stream(
        weakref.proxy(ctx), handle, finalizer=None
    )

    return nb_stream


def _numba_predict(alpha, x_in, F, P, Q):

    x, y, z = cuda.grid(3)
    _, _, strideZ = cuda.gridsize(3)
    tz = cuda.threadIdx.z

    dim_x = P.shape[0]

    xx_block_size = dim_x * dim_x * cuda.blockDim.z
    xx_idx = dim_x * dim_x * tz

    s_buffer = cuda.shared.array(shape=0, dtype=float32)

    s_A = s_buffer[: (xx_block_size * 1)]
    s_F = s_buffer[(xx_block_size * 1) :]

    x_key = x * dim_x + y

    #  Each i is a different point
    for z_idx in range(z, x_in.shape[2], strideZ):

        s_F[xx_idx + x_key] = F[x, y, z_idx]

        cuda.syncthreads()

        #  Load alpha_sq and Q into registers
        alpha_sq = alpha[0, 0, z_idx]
        local_Q = Q[x, y, z_idx]

        #  Compute new self.x
        temp: x_in.dtype = 0
        if y == 0:
            for j in range(dim_x):
                temp += s_F[xx_idx + (x * dim_x + j)] * x_in[j, y, z_idx]

            x_in[x, y, z_idx] = temp

        #  Compute dot(self.F, self.P)
        temp: x_in.dtype = 0
        for j in range(dim_x):
            temp += s_F[xx_idx + (x * dim_x + j)] * P[j, y, z_idx]

        s_A[xx_idx + x_key] = temp

        cuda.syncthreads()

        #  Compute dot(dot(self.F, self.P), self.F.T)
        temp: x_in.dtype = 0
        for j in range(dim_x):
            temp += (
                s_A[xx_idx + (x * dim_x + j)] * s_F[xx_idx + (y * dim_x + j)]
            )

        #  Compute alpha^2 * dot(dot(self.F, self.P), self.F.T) + self.Q
        P[x, y, z_idx] = alpha_sq * temp + local_Q


def _numba_update(x_in, z_in, H, P, R):

    x, y, z = cuda.grid(3)
    _, _, strideZ = cuda.gridsize(3)
    tz = cuda.threadIdx.z

    dim_x = P.shape[0]
    dim_z = R.shape[0]

    xx_block_size = dim_x * dim_x * cuda.blockDim.z
    xz_block_size = dim_x * dim_z * cuda.blockDim.z
    zz_block_size = dim_z * dim_z * cuda.blockDim.z
    xx_idx = dim_x * dim_x * tz
    xz_idx = dim_x * dim_z * tz
    zz_idx = dim_z * dim_z * tz

    s_buffer = cuda.shared.array(shape=0, dtype=float32)

    s_A = s_buffer[: (xx_block_size * 1)]
    s_B = s_buffer[(xx_block_size * 1) : (xx_block_size * 2)]
    s_P = s_buffer[(xx_block_size * 2) : (xx_block_size * 3)]
    s_H = s_buffer[(xx_block_size * 3) : (xx_block_size * 3 + xz_block_size)]
    s_K = s_buffer[
        (xx_block_size * 3 + xz_block_size) : (
            xx_block_size * 3 + xz_block_size * 2
        )
    ]
    s_R = s_buffer[
        (xx_block_size * 3 + xz_block_size * 2) : (
            xx_block_size * 3 + xz_block_size * 2 + zz_block_size
        )
    ]
    s_y = s_buffer[(xx_block_size * 3 + xz_block_size * 2 + zz_block_size) :]

    x_key = x * dim_x + y
    z_key = x * dim_z + y

    #  Each i is a different point
    for z_idx in range(z, x_in.shape[2], strideZ):

        s_P[xx_idx + x_key] = P[x, y, z_idx]

        if x < dim_z:
            s_H[xz_idx + x_key] = H[x, y, z_idx]

        if x < dim_z and y < dim_z:
            s_R[zz_idx + z_key] = R[x, y, z_idx]

        cuda.syncthreads()

        #  Compute self.y : z = dot(self.H, self.x)
        temp: x_in.dtype = 0.0
        if x < dim_z and y == 0:
            temp_z: x_in.dtype = z_in[x, y, z_idx]
            for j in range(dim_x):
                temp += s_H[xz_idx + (x * dim_x + j)] * x_in[j, y, z_idx]

            s_y[(dim_z * tz) + x] = temp_z - temp

        #  Compute PHT : dot(self.P, self.H.T)
        temp: x_in.dtype = 0.0
        if y < 2:
            for j in range(dim_x):
                temp += (
                    s_P[xx_idx + (x * dim_x + j)]
                    * s_H[xz_idx + (y * dim_x + j)]
                )

            #  s_A holds PHT
            s_A[xx_idx + z_key] = temp

        cuda.syncthreads()

        #  Compute self.S : dot(self.H, PHT) + self.R
        temp: x_in.dtype = 0.0
        if x < dim_z and y < dim_z:
            for j in range(dim_x):
                temp += (
                    s_H[xz_idx + (x * dim_x + j)]
                    * s_A[xx_idx + (j * dim_z + y)]
                )

            #  s_B holds S - system uncertainty
            s_B[xx_idx + z_key] = temp + s_R[zz_idx + z_key]

        cuda.syncthreads()

        if x < dim_z and y < dim_z:

            #  Compute linalg.inv(S)
            #  Hardcoded for 2x2
            sign = 1 if (x + y) % 2 == 0 else -1

            #  sign * determinant
            sign_det = sign * (
                (s_B[xx_idx + (0 * dim_z + 0)] * s_B[xx_idx + (1 * dim_z + 1)])
                - (
                    s_B[xx_idx + (1 * dim_z + 0)]
                    * s_B[xx_idx + (0 * dim_z + 1)]
                )
            )

            #  s_B hold SI - inverse system uncertainty
            temp = s_B[xx_idx + ((1 - x) * dim_z + (1 - y))] / sign_det
            s_B[xx_idx + z_key] = temp

        cuda.syncthreads()

        #  Compute self.K : dot(PHT, self.SI)
        #  kalman gain
        temp: x_in.dtype = 0.0
        if y < 2:
            for j in range(dim_z):
                temp += (
                    s_A[xx_idx + (x * dim_z + j)]
                    * s_B[xx_idx + (y * dim_z + j)]
                )

            s_K[xz_idx + z_key] = temp

        cuda.syncthreads()

        #  Compute self.x : self.x + cp.dot(self.K, self.y)
        temp: x_in.dtype = 0.0
        if y == 0:
            for j in range(dim_z):
                temp += s_K[xz_idx + (x * dim_z + j)] * s_y[(dim_z * tz) + j]

            x_in[x, y, z_idx] += temp

        #  Compute I_KH = self_I - dot(self.K, self.H)
        temp: x_in.dtype = 0.0
        for j in range(dim_z):
            temp += (
                s_K[xz_idx + (x * dim_z + j)] * s_H[xz_idx + (j * dim_x + y)]
            )

        #  s_A holds I_KH
        s_A[xx_idx + x_key] = (1.0 if x == y else 0.0) - temp

        cuda.syncthreads()

        #  Compute self.P = dot(dot(I_KH, self.P), I_KH.T) +
        #  dot(dot(self.K, self.R), self.K.T)

        #  Compute dot(I_KH, self.P)
        temp: x_in.dtype = 0.0
        for j in range(dim_x):
            temp += (
                s_A[xx_idx + (x * dim_x + j)] * s_P[xx_idx + (j * dim_x + y)]
            )

        #  s_A holds dot(I_KH, self.P)
        s_B[xx_idx + x_key] = temp

        cuda.syncthreads()

        #  Compute dot(dot(I_KH, self.P), I_KH.T)
        temp: x_in.dtype = 0.0
        for j in range(dim_x):
            temp += (
                s_B[xx_idx + (x * dim_x + j)] * s_A[xx_idx + (y * dim_x + j)]
            )

        #  Compute dot(self.K, self.R)
        temp2: x_in.dtype = 0.0
        if y < dim_z:
            for j in range(dim_z):
                temp2 += (
                    s_K[xz_idx + (x * dim_z + j)]
                    * s_R[zz_idx + (j * dim_z + y)]
                )

        #  s_A holds dot(self.K, self.R)
        s_A[xx_idx + z_key] = temp2

        cuda.syncthreads()

        #  Compute dot(dot(self.K, self.R), self.K.T)
        temp2: x_in.dtype = 0.0
        for j in range(dim_z):
            temp2 += (
                s_A[xx_idx + (x * dim_z + j)] * s_K[xz_idx + (y * dim_z + j)]
            )

        P[x, y, z_idx] = temp + temp2


def _numba_kalman_signature(ty):
    return void(
        ty[:, :, :], ty[:, :, :], ty[:, :, :], ty[:, :, :], ty[:, :, :],
    )


def _get_backend_kernel(dtype, grid, block, smem, stream, use_numba, k_type):

    nb_stream = stream_cupy_to_numba(stream)
    kernel = _numba_kernel_cache[(dtype.name, k_type)]

    if kernel:
        return kernel[grid, block, nb_stream, smem]
    else:
        raise ValueError(
            "Kernel {} not found in _numba_kernel_cache".format(k_type)
        )


def _populate_kernel_cache(np_type, use_numba, k_type):

    # Check in np_type is a supported option
    try:
        numba_type, _ = _SUPPORTED_TYPES[np_type]

    except ValueError:
        raise Exception("No kernel found for datatype {}".format(np_type))

    if (str(numba_type), k_type) in _numba_kernel_cache:
        return

    sig = _numba_kalman_signature(numba_type)
    if k_type == GPUKernel.PREDICT:
        _numba_kernel_cache[(str(numba_type), k_type)] = cuda.jit(
            sig, fastmath=True
        )(_numba_predict)
    else:
        _numba_kernel_cache[(str(numba_type), k_type)] = cuda.jit(
            sig, fastmath=True
        )(_numba_update)


class KalmanFilter(object):

    #  documentation
    def __init__(self, num_points, dim_x, dim_z, dim_u=0):

        self.num_points = num_points

        if dim_x < 1:
            raise ValueError("dim_x must be 1 or greater")
        if dim_z < 1:
            raise ValueError("dim_z must be 1 or greater")
        if dim_u < 0:
            raise ValueError("dim_u must be 0 or greater")

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        # 1. if read-only and same initial, we can have one copy
        # 2. if not read-only and same initial, use broadcasting
        self.x = cp.zeros(
            (dim_x, 1, self.num_points), dtype=cp.float32
        )  # state

        self.P = cp.repeat(
            cp.identity(dim_x, dtype=cp.float32)[:, :, np.newaxis],
            self.num_points,
            axis=2,
        )  # uncertainty covariance

        self.Q = cp.repeat(
            cp.identity(dim_x, dtype=cp.float32)[:, :, np.newaxis],
            self.num_points,
            axis=2,
        )  # process uncertainty

        # self.B = None  # control transition matrix

        self.F = cp.repeat(
            cp.identity(dim_x, dtype=cp.float32)[:, :, np.newaxis],
            self.num_points,
            axis=2,
        )  # state transition matrix

        self.H = cp.zeros(
            (dim_z, dim_z, self.num_points), dtype=cp.float32
        )  # Measurement function

        self.R = cp.repeat(
            cp.identity(dim_z, dtype=cp.float32)[:, :, np.newaxis],
            self.num_points,
            axis=2,
        )  # process uncertainty

        self._alpha_sq = cp.ones(
            (1, 1, self.num_points), dtype=cp.float32
        )  # fading memory control

        self.M = cp.zeros(
            (dim_z, dim_z, self.num_points), dtype=cp.float32
        )  # process-measurement cross correlation

        self.z = cp.empty((dim_z, 1, self.num_points), dtype=cp.float32)

        _populate_kernel_cache(self.x.dtype.type, True, GPUKernel.PREDICT)
        _populate_kernel_cache(self.x.dtype.type, True, GPUKernel.UPDATE)

    def predict(self):
        d = cp.cuda.device.Device(0)
        numSM = d.attributes["MultiProcessorCount"]
        threadsperblock = (self.dim_x, self.dim_x, 16)
        blockspergrid = (1, 1, numSM * 20)

        A_size = self.dim_x * self.dim_x
        F_size = self.dim_x * self.dim_x

        total_size = A_size + F_size

        shared_mem_size = (
            total_size * threadsperblock[2] * self.x.dtype.itemsize
        )

        kernel = _get_backend_kernel(
            self.x.dtype,
            blockspergrid,
            threadsperblock,
            shared_mem_size,
            cp.cuda.stream.Stream(null=True),
            True,
            GPUKernel.PREDICT,
        )

        kernel(self._alpha_sq, self.x, self.F, self.P, self.Q,)

    def update(self):
        d = cp.cuda.device.Device(0)
        numSM = d.attributes["MultiProcessorCount"]
        threadsperblock = (self.dim_x, self.dim_x, 16)
        blockspergrid = (1, 1, numSM * 20)

        A_size = self.dim_x * self.dim_x
        B_size = self.dim_x * self.dim_x
        P_size = self.dim_x * self.dim_x
        H_size = self.dim_z * self.dim_x
        K_size = self.dim_x * self.dim_z
        R_size = self.dim_z * self.dim_z
        y_size = self.dim_z * 1

        total_size = (
            A_size + B_size + P_size + H_size + K_size + R_size + y_size
        )

        shared_mem_size = (
            total_size * threadsperblock[2] * self.x.dtype.itemsize
        )

        kernel = _get_backend_kernel(
            self.x.dtype,
            blockspergrid,
            threadsperblock,
            shared_mem_size,
            cp.cuda.stream.Stream(null=True),
            True,
            GPUKernel.UPDATE,
        )

        kernel(self.x, self.z, self.H, self.P, self.R,)
