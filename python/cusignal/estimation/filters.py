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
import numpy as np

from numba import cuda, float32


@cuda.jit(fastmath=False)
def _numba_predict(num, dim_x, alpha, x_in, F, P, Q):

    x, y, z = cuda.grid(3)
    _, _, strideZ = cuda.gridsize(3)
    tz = cuda.threadIdx.z

    s_data = cuda.shared.array(shape=0, dtype=float32)

    #  Each i is a different point
    for z_idx in range(z, num, strideZ):

        #  Compute new self.x
        temp: x_in.dtype = 0
        if y == 0:
            for j in range(dim_x):
                temp += F[x, j, z_idx] * x_in[j, y, z_idx]

            x_in[x, 0, z_idx] = temp

        #  Compute dot(self.F, self.P)
        temp: x_in.dtype = 0
        for j in range(dim_x):
            temp += F[x, j, z_idx] * P[j, y, z_idx]

        s_data[(dim_x * dim_x * tz) + (x * dim_x + y)] = temp

        cuda.syncthreads()

        #  Compute dot(dot(self.F, self.P), self.F.T)
        temp: x_in.dtype = 0
        for j in range(dim_x):
            temp += (
                s_data[(dim_x * dim_x * tz) + (x * dim_x + j)] * F[y, j, z_idx]
            )

        #  Compute (alpha * alpha) * dot(dot(self.F, self.P), self.F.T)
        temp *= alpha[0, 0, z_idx] * alpha[0, 0, z_idx]

        #  Compute
        #  (alpha * alpha) * dot(dot(self.F, self.P), self.F.T) + self.Q
        P[x, y, z_idx] = temp + Q[x, y, z_idx]


@cuda.jit(fastmath=True)
def _numba_update(num, dim_x, dim_z, x_in, z_in, H, P, R, S, SI, K, y_in):

    x, y, z = cuda.grid(3)
    _, _, strideZ = cuda.gridsize(3)
    tz = cuda.threadIdx.z

    s_buffer = cuda.shared.array(shape=0, dtype=float32)

    s_A = s_buffer[: (dim_x * dim_x * cuda.blockDim.z)]
    s_I_KH = s_buffer[(dim_x * dim_x * cuda.blockDim.z) :]

    #  Each i is a different point
    for z_idx in range(z, num, strideZ):

        #  Compute self.y : z = dot(self.H, self.x)
        temp: x_in.dtype = 0.0
        if x < dim_z and y == 0:
            for j in range(dim_x):
                temp += H[x, j, z_idx] * x_in[j, y, z_idx]

            y_in[x, y, z_idx] = z_in[x, y, z_idx] - temp

        #  Compute PHT : dot(self.P, self.H.T)
        temp: x_in.dtype = 0.0
        if y < 2:
            for j in range(dim_x):
                temp += P[x, j, z_idx] * H[y, j, z_idx]

            s_A[(dim_x * dim_x * tz) + (x * dim_z + y)] = temp

        cuda.syncthreads()

        #  Compute self.S : dot(self.H, PHT) + self.R
        temp: x_in.dtype = 0.0
        if x < dim_z and y < dim_z:
            for j in range(dim_x):
                temp += (
                    H[x, j, z_idx]
                    * s_A[(dim_x * dim_x * tz) + (j * dim_z + y)]
                )

            S[x, y, z_idx] = temp + R[x, y, z_idx]

            #  Still work with dim_z
            #  Compute linalg.inv(S)
            det = (S[0, 0, z_idx] * S[1, 1, z_idx]) - (
                S[1, 0, z_idx] * S[0, 1, z_idx]
            )
            sign = 1 if (x + y) % 2 == 0 else -1

            SI[x, y, z_idx] = S[1 - x, 1 - y, z_idx] / (sign * det)

        #  Compute self.K : dot(PHT, self.SI)
        temp: x_in.dtype = 0.0
        if y < 2:
            for j in range(dim_z):
                temp += (
                    s_A[(dim_x * dim_x * tz) + (x * dim_z + j)]
                    * SI[j, y, z_idx]
                )

            K[x, y, z_idx] = temp

        #  Compute self.x : self.x + cp.dot(self.K, self.y)
        temp: x_in.dtype = 0.0
        if y == 0:
            for j in range(dim_z):
                temp += K[x, j, z_idx] * y_in[j, y, z_idx]
            x_in[x, y, z_idx] += temp

        #  Compute I_KH = self_I - dot(self.K, self.H)
        temp: x_in.dtype = 0.0
        for j in range(dim_z):
            temp += K[x, j, z_idx] * H[j, y, z_idx]

        s_I_KH[(dim_x * dim_x * tz) + (x * dim_x + y)] = (
            1.0 if x == y else 0.0
        ) - temp

        cuda.syncthreads()

        #  Compute self.P = dot(dot(I_KH, self.P), I_KH.T) +
        #  dot(dot(self.K, self.R), self.K.T)

        #  Compute dot(I_KH, self.P)
        temp: x_in.dtype = 0.0
        for j in range(dim_x):
            temp += (
                s_I_KH[(dim_x * dim_x * tz) + (x * dim_x + j)]
                * P[j, y, z_idx]
            )

        s_A[(dim_x * dim_x * tz) + (x * dim_x + y)] = temp

        cuda.syncthreads()

        #  Compute dot(dot(I_KH, self.P), I_KH.T)
        #  Where dot(I_KH, self.P) equals
        #  s_data[(dim_x * dim_x * tz) + (x * dim_x + y)]
        temp2: x_in.dtype = 0.0
        for j in range(dim_x):
            temp2 += (
                s_A[(dim_x * dim_x * tz) + (x * dim_x + j)]
                * s_I_KH[(dim_x * dim_x * tz) + (y * dim_x + j)]
            )

        #  Compute dot(self.K, self.R)
        temp: x_in.dtype = 0.0
        if y < dim_z:
            for j in range(dim_z):
                temp += K[x, j, z_idx] * R[j, y, z_idx]

        s_A[(dim_x * dim_x * tz) + (x * dim_z + y)] = temp

        cuda.syncthreads()

        #  Compute dot(dot(self.K, self.R), self.K.T)
        temp: x_in.dtype = 0.0
        for j in range(dim_z):
            temp += (
                s_A[(dim_x * dim_x * tz) + (x * dim_z + j)]
                * K[y, j, z_idx]
            )

        P[x, y, z_idx] = temp + temp2


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

        self.K = cp.zeros(
            (dim_x, dim_z, self.num_points), dtype=cp.float32
        )  # kalman gain

        self.y = cp.zeros((dim_z, 1, self.num_points), dtype=cp.float32)

        self.S = cp.zeros(
            (dim_z, dim_z, self.num_points), dtype=cp.float32
        )  # system uncertainty

        self.SI = cp.zeros(
            (dim_z, dim_z, self.num_points), dtype=cp.float32
        )  # inverse system uncertainty

    def predict(self):
        d = cp.cuda.device.Device(0)
        numSM = d.attributes["MultiProcessorCount"]
        threadsperblock = (self.dim_x, self.dim_x, 16)
        blockspergrid = (1, 1, numSM * 20)

        shared_mem_size = (
            self.dim_x
            * self.dim_x
            * threadsperblock[2]
            * self.x.dtype.itemsize
        )

        _numba_predict[blockspergrid, threadsperblock, 0, shared_mem_size](
            self.num_points,
            self.dim_x,
            self._alpha_sq,
            self.x,
            self.F,
            self.P,
            self.Q,
        )

    def update(self):
        d = cp.cuda.device.Device(0)
        numSM = d.attributes["MultiProcessorCount"]
        threadsperblock = (self.dim_x, self.dim_x, 16)
        blockspergrid = (1, 1, numSM * 20)

        shared_mem_size = (
            self.dim_x
            * self.dim_x
            * threadsperblock[2]
            * 2
            * self.x.dtype.itemsize
        )

        _numba_update[blockspergrid, threadsperblock, 0, shared_mem_size](
            self.num_points,
            self.dim_x,
            self.dim_z,
            self.x,
            self.z,
            self.H,
            self.P,
            self.R,
            self.S,
            self.SI,
            self.K,
            self.y,
        )
