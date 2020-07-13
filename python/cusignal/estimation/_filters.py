# Copyright (c) 2019-2020, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cupy as cp

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
    cp.float32: [float32, "float"],
    cp.float64: [float64, "double"],
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

    _, _, tz = cuda.grid(3)
    _, _, strideZ = cuda.gridsize(3)

    lty = cuda.threadIdx.x
    ltx = cuda.threadIdx.y
    ltz = cuda.threadIdx.z

    dim_x = P.shape[1]

    s_XX_A = cuda.shared.array(shape=(16, 4, 4), dtype=float64)
    s_XX_F = cuda.shared.array(shape=(16, 4, 4), dtype=float64)
    s_XX_P = cuda.shared.array(shape=(16, 4, 4), dtype=float64)

    #  Each i is a different point
    for gtz in range(tz, x_in.shape[0], strideZ):

        s_XX_F[ltz, lty, ltx] = F[
            gtz, lty, ltx,
        ]

        s_XX_P[ltz, lty, ltx] = P[
            gtz, lty, ltx,
        ]

        cuda.syncthreads()

        #  Load alpha_sq and Q into registers
        alpha_sq = alpha[gtz, 0, 0]
        local_Q = Q[gtz, lty, ltx]

        #  Compute new self.x
        #  x_in = 4x1
        temp: x_in.dtype = 0
        if ltx == 0:  # ltx = tx
            for j in range(dim_x):
                temp += s_XX_F[ltz, lty, j] * x_in[gtz, j, ltx]

            x_in[gtz, lty, ltx] = temp

        #  Compute dot(self.F, self.P)
        temp: x_in.dtype = 0
        for j in range(dim_x):
            temp += s_XX_F[ltz, lty, j] * s_XX_P[ltz, j, ltx]

        s_XX_A[ltz, lty, ltx] = temp

        cuda.syncthreads()

        #  Compute dot(dot(self.F, self.P), self.F.T)
        temp: x_in.dtype = 0
        for j in range(dim_x):
            temp += s_XX_A[ltz, lty, j] * s_XX_F[ltz, ltx, j]

        #  Compute alpha^2 * dot(dot(self.F, self.P), self.F.T) + self.Q
        P[gtz, lty, ltx] = alpha_sq * temp + local_Q


def _numba_update(x_in, z_in, H, P, R):

    _, _, btz = cuda.grid(3)
    _, _, strideZ = cuda.gridsize(3)

    lty = cuda.threadIdx.x
    ltx = cuda.threadIdx.y
    ltz = cuda.threadIdx.z

    dim_x = P.shape[1]
    dim_z = R.shape[1]

    s_XX_A = cuda.shared.array(shape=(16, 4, 4), dtype=float64)
    s_XX_B = cuda.shared.array(shape=(16, 4, 4), dtype=float64)
    s_XX_P = cuda.shared.array(shape=(16, 4, 4), dtype=float64)
    s_ZX_H = cuda.shared.array(shape=(16, 2, 4), dtype=float64)
    s_XZ_K = cuda.shared.array(shape=(16, 4, 2), dtype=float64)
    s_XZ_A = cuda.shared.array(shape=(16, 4, 2), dtype=float64)
    s_ZZ_A = cuda.shared.array(shape=(16, 2, 2), dtype=float64)
    s_ZZ_R = cuda.shared.array(shape=(16, 2, 2), dtype=float64)
    s_Z1_y = cuda.shared.array(shape=(16, 2, 1), dtype=float64)

    #  Each i is a different point
    for gtz in range(btz, x_in.shape[0], strideZ):

        s_XX_P[ltz, lty, ltx] = P[
            gtz, lty, ltx,
        ]

        if lty < dim_z:
            s_ZX_H[ltz, lty, ltx] = H[gtz, lty, ltx]

        if lty < dim_z and ltx < dim_z:
            s_ZZ_R[ltz, lty, ltx] = R[gtz, lty, ltx]

        cuda.syncthreads()

        #  Compute self.y : z = dot(self.H, self.x) --> Z1
        temp: x_in.dtype = 0.0
        if lty < dim_z and ltx == 0:
            temp_z: x_in.dtype = z_in[gtz, lty, ltx]
            for j in range(dim_x):
                temp += s_ZX_H[ltz, lty, j] * x_in[gtz, j, ltx]

            s_Z1_y[ltz, lty, ltx] = temp_z - temp

        #  Compute PHT : dot(self.P, self.H.T) --> XZ
        temp: x_in.dtype = 0.0
        if ltx < dim_z:
            for j in range(dim_x):
                temp += s_XX_P[ltz, lty, j] * s_ZX_H[ltz, ltx, j]

            #  s_XX_A holds PHT
            s_XZ_A[ltz, lty, ltx] = temp

        cuda.syncthreads()

        #  Compute self.S : dot(self.H, PHT) + self.R --> ZZ
        temp: x_in.dtype = 0.0
        if lty < dim_z and ltx < dim_z:
            for j in range(dim_x):
                temp += s_ZX_H[ltz, lty, j] * s_XZ_A[ltz, j, ltx]

            #  s_XX_B holds S - system uncertainty
            s_ZZ_A[ltz, lty, ltx] = temp + s_ZZ_R[ltz, lty, ltx]

        cuda.syncthreads()

        if lty < dim_z and ltx < dim_z:

            #  Compute linalg.inv(S)
            #  Hardcoded for 2x2
            sign = 1 if (lty + ltx) % 2 == 0 else -1

            #  sign * determinant
            sign_det = float32(sign) * (
                (s_ZZ_A[ltz, 0, 0] * s_ZZ_A[ltz, 1, 1])
                - (s_ZZ_A[ltz, 1, 0] * s_ZZ_A[ltz, 0, 1])
            )

            #  s_ZZ_A hold SI - inverse system uncertainty
            temp = s_ZZ_A[ltz, 1 - lty, 1 - ltx] / sign_det

        cuda.syncthreads()

        if lty < dim_z and ltx < dim_z:
            s_ZZ_A[ltz, lty, ltx] = temp

        cuda.syncthreads()

        #  Compute self.K : dot(PHT, self.SI) --> ZZ
        #  kalman gain
        temp: x_in.dtype = 0.0
        if ltx < dim_z:
            for j in range(dim_z):
                temp += (
                    s_XZ_A[ltz, lty, j]
                    * s_ZZ_A[ltz, ltx, j]
                )
            s_XZ_K[ltz, lty, ltx] = temp

        cuda.syncthreads()

        #  Compute self.x : self.x + cp.dot(self.K, self.ltx) --> X1
        temp: x_in.dtype = 0.0
        if ltx == 0:
            for j in range(dim_z):
                temp += s_XZ_K[ltz, lty, j] * s_Z1_y[ltz, j, ltx]

            x_in[gtz, lty, ltx] += temp

        #  Compute I_KH = self_I - dot(self.K, self.H) --> XX
        temp: x_in.dtype = 0.0
        for j in range(dim_z):
            temp += s_XZ_K[ltz, lty, j] * s_ZX_H[ltz, j, ltx]

        #  s_XX_A holds I_KH
        s_XX_A[ltz, lty, ltx] = (1.0 if lty == ltx else 0.0) - temp

        cuda.syncthreads()

        #  Compute self.P = dot(dot(I_KH, self.P), I_KH.T) +
        #  dot(dot(self.K, self.R), self.K.T)

        #  Compute dot(I_KH, self.P) --> XX
        temp: x_in.dtype = 0.0
        for j in range(dim_x):
            temp += s_XX_A[ltz, lty, j] * s_XX_P[ltz, j, ltx]

        #  s_XX_B holds dot(I_KH, self.P)
        s_XX_B[ltz, lty, ltx] = temp

        cuda.syncthreads()

        #  Compute dot(dot(I_KH, self.P), I_KH.T) --> XX
        temp: x_in.dtype = 0.0
        for j in range(dim_x):
            temp += s_XX_B[ltz, lty, j] * s_XX_A[ltz, ltx, j]

        s_XX_P[ltz, lty, ltx] = temp

        cuda.syncthreads()

        #  Compute dot(self.K, self.R) --> XZ
        temp: x_in.dtype = 0.0
        if ltx < dim_z:
            for j in range(dim_z):
                temp += s_XZ_K[ltz, lty, j] * s_ZZ_R[ltz, j, ltx]

            #  s_XX_A holds dot(self.K, self.R)
            s_XZ_A[ltz, lty, ltx] = temp

        cuda.syncthreads()

        #  Compute dot(dot(self.K, self.R), self.K.T) --> XX
        temp: x_in.dtype = 0.0
        for j in range(dim_z):
            temp += s_XZ_A[ltz, lty, j] * s_XZ_K[ltz, ltx, j]

        P[gtz, lty, ltx] = temp + s_XX_P[ltz, lty, ltx]


def _numba_kalman_signature(ty):
    return void(
        ty[:, :, :], ty[:, :, :], ty[:, :, :], ty[:, :, :], ty[:, :, :],
    )


# Custom Cupy raw kernel
# Matthew Nicely - mnicely@nvidia.com
cuda_code = """
// Compute linalg.inv(S)
template<typename T, int DIM_Z>
__device__ T inverse(
    const int & ltx,
    const int & lty,
    const int & ltz,
    T(&s_ZZ_A)[16][DIM_Z][DIM_Z],
    T(&s_ZZ_I)[16][DIM_Z][DIM_Z]) {

    T temp {};

    // Interchange the row of matrix
    if ( lty == 0 && ltx < DIM_Z) {
#pragma unroll ( DIM_Z - 1 )
        for ( int i = DIM_Z - 1; i > 0; i-- ) {
            if ( s_ZZ_A[ltz][i - 1][0] < s_ZZ_A[ltz][i][0] ) {
                    temp = s_ZZ_A[ltz][i][ltx];
                    s_ZZ_A[ltz][i][ltx] = s_ZZ_A[ltz][i - 1][ltx];
                    s_ZZ_A[ltz][i - 1][ltx] = temp;

                    temp = s_ZZ_I[ltz][i][ltx];
                    s_ZZ_I[ltz][i][ltx] = s_ZZ_I[ltz][i - 1][ltx];
                    s_ZZ_I[ltz][i - 1][ltx] = temp;
            }
        }
    }

    if ( lty < DIM_Z && ltx < DIM_Z ) {

        // Replace a row by sum of itself and a
        // constant multiple of another row of the matrix
#pragma unroll DIM_Z
        for ( int i = 0; i < DIM_Z; i++ ) {
            T temp2 = s_ZZ_I[ltz][i][ltx];

            if ( lty != i ) {
                temp = s_ZZ_A[ltz][lty][i] / s_ZZ_A[ltz][i][i];
                s_ZZ_A[ltz][lty][ltx] -= s_ZZ_A[ltz][i][ltx] * temp;
                s_ZZ_I[ltz][lty][ltx] -= s_ZZ_I[ltz][i][ltx] * temp;
            }
        }

        // Multiply each row by a nonzero integer.
        // Divide row element by the diagonal element
        temp = s_ZZ_A[ltz][lty][lty];
        s_ZZ_A[ltz][lty][ltx] = s_ZZ_A[ltz][lty][ltx] / temp;
        s_ZZ_I[ltz][lty][ltx] = s_ZZ_I[ltz][lty][ltx] / temp;
    }

    return ( s_ZZ_I[ltz][lty][ltx] );
}


template<typename T, int DIM_X, int MAX_TPB, int MIN_BPSM>
__global__ void __launch_bounds__(MAX_TPB) _cupy_predict(
        const int num_points,
        const T * __restrict__ alpha_sq,
        T * __restrict__ x_in,
        const T * __restrict__ F,
        T * __restrict__ P,
        const T * __restrict__ Q
        ) {

    __shared__ T s_XX_A[16+1][DIM_X][DIM_X+1];
    __shared__ T s_XX_F[16+1][DIM_X][DIM_X+1];
    __shared__ T s_XX_P[16+1][DIM_X][DIM_X];

    const auto ltx = threadIdx.x;
    const auto lty = threadIdx.y;
    const auto ltz = threadIdx.z;

    const int btz { static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z) };

    const int stride_z { static_cast<int>( blockDim.z * gridDim.z ) };

    const int x_value { lty * DIM_X + ltx };

    for ( int gtz = btz; gtz < num_points; gtz += stride_z ) {

        s_XX_F[ltz][lty][ltx] = F[gtz * DIM_X * DIM_X + x_value];

        __syncthreads();

        T alpha2 { alpha_sq[gtz] };
        T localQ { Q[gtz * DIM_X * DIM_X + x_value] };
        T localP { P[gtz * DIM_X * DIM_X + x_value] };

        T temp {};

        // Compute self.x = dot(F, self.x)
        if ( ltx == 0 ) {
#pragma unroll DIM_X
            for ( int j = 0; j < DIM_X; j++ ) {
                temp += s_XX_F[ltz][lty][j] *
                    x_in[gtz * DIM_X * 1 + j * 1 + ltx];
            }
            x_in[gtz * DIM_X * 1 + lty * 1 + ltx]   = temp;
        }

        s_XX_P[ltz][lty][ltx] = localP;

        __syncthreads();

        // Compute dot(F, self.P)
        temp = 0.0;
#pragma unroll DIM_X
        for ( int j = 0; j < DIM_X; j++ ) {
            temp += s_XX_F[ltz][lty][j] *
                s_XX_P[ltz][j][ltx];
        }
        s_XX_A[ltz][lty][ltx] = temp;

        __syncthreads();

        // Compute dot(dot(F, self.P), F.T)
        temp = 0.0;
#pragma unroll DIM_X
        for ( int j = 0; j < DIM_X; j++ ) {
            temp += s_XX_A[ltz][lty][j] *
                s_XX_F[ltz][ltx][j];
        }

        // Compute self._alpha_sq * dot(dot(F, self.P), F.T) + Q
        // Where temp = dot(dot(F, self.P), F.T)
        P[gtz * DIM_X * DIM_X + x_value] =
            alpha2 * temp + localQ;
    }
}


template<typename T, int DIM_X, int DIM_Z, int MAX_TPB, int MIN_BPSM>
__global__ void __launch_bounds__(MAX_TPB) _cupy_update(
        const int num_points,
        T * __restrict__ x_in,
        const T * __restrict__ z_in,
        const T * __restrict__ H,
        T * __restrict__ P,
        const T * __restrict__ R
        ) {

    __shared__ T s_XX_A[16][DIM_X][DIM_X+1];
    __shared__ T s_XX_B[16][DIM_X][DIM_X+1];
    __shared__ T s_XX_P[16][DIM_X][DIM_X];
    __shared__ T s_ZX_H[16][DIM_Z][DIM_X];
    __shared__ T s_XZ_K[16][DIM_X][DIM_Z];
    __shared__ T s_XZ_A[16][DIM_X][DIM_Z];
    __shared__ T s_ZZ_A[16][DIM_Z][DIM_Z];
    __shared__ T s_ZZ_R[16][DIM_Z][DIM_Z];
    __shared__ T s_ZZ_I[16][DIM_Z][DIM_Z];
    __shared__ T s_Z1_y[16][DIM_Z][1];

    const auto ltx = threadIdx.x;
    const auto lty = threadIdx.y;
    const auto ltz = threadIdx.z;

    const int btz {
        static_cast<int>( blockIdx.z * blockDim.z + threadIdx.z ) };

    const int stride_z { static_cast<int>( blockDim.z * gridDim.z ) };

    const int x_value { lty * DIM_X + ltx };
    const int z_value { lty * DIM_Z + ltx };

    for ( int gtz = btz; gtz < num_points; gtz += stride_z ) {

        if ( lty < DIM_Z ) {
            s_ZX_H[ltz][lty][ltx] =
                H[gtz * DIM_Z * DIM_X + x_value];
        }

        __syncthreads();

        s_XX_P[ltz][lty][ltx] = P[gtz * DIM_X * DIM_X + x_value];

        if ( ( lty < DIM_Z ) && ( ltx < DIM_Z ) ) {
            s_ZZ_R[ltz][lty][ltx] =
                R[gtz * DIM_Z * DIM_Z + z_value];

            if ( lty == ltx ) {
                s_ZZ_I[ltz][lty][ltx] = 1.0;
            } else {
                s_ZZ_I[ltz][lty][ltx] = 0.0;
            }
        }

        T temp {};

        // Compute self.y : z = dot(self.H, self.x) --> Z1
        if ( ( ltx == 0 ) && ( lty < DIM_Z ) ) {
            T temp_z { z_in[gtz * DIM_Z + lty] };

#pragma unroll DIM_X
            for ( int j = 0; j < DIM_X; j++ ) {
                temp += s_ZX_H[ltz][lty][j] *
                    x_in[gtz * DIM_X + j];
            }

            s_Z1_y[ltz][lty][ltx] = temp_z - temp;
        }

        __syncthreads();

        // Compute PHT : dot(self.P, self.H.T) --> XZ
        temp = 0.0;
        if ( ltx < DIM_Z ) {
#pragma unroll DIM_X
            for ( int j = 0; j < DIM_X; j++ ) {
                temp += s_XX_P[ltz][lty][j] *
                    s_ZX_H[ltz][ltx][j];
            }
            // s_XX_A holds PHT
            s_XZ_A[ltz][lty][ltx] = temp;
        }

        __syncthreads();

        // Compute self.S : dot(self.H, PHT) + self.R --> ZZ
        temp = 0.0;
        if ( ( ltx < DIM_Z ) && ( lty < DIM_Z ) ) {
#pragma unroll DIM_X
            for ( int j = 0; j < DIM_X; j++ ) {
                temp += s_ZX_H[ltz][lty][j] *
                    s_XZ_A[ltz][j][ltx];
            }
            // s_XX_B holds S - system uncertainty
            s_ZZ_A[ltz][lty][ltx] = temp + s_ZZ_R[ltz][lty][ltx];
        }

        __syncthreads();

        // Compute matrix inversion
        temp = inverse(ltx, lty, ltz, s_ZZ_A, s_ZZ_I);

        __syncthreads();

        if ( ( ltx < DIM_Z ) && ( lty < DIM_Z ) ) {
            // s_XX_B hold SI - inverse system uncertainty
            s_ZZ_A[ltz][lty][ltx] = temp;
        }

        __syncthreads();

        //  Compute self.K : dot(PHT, self.SI) --> ZZ
        //  kalman gain
        temp = 0.0;
        if ( ltx < DIM_Z ) {
#pragma unroll DIM_Z
            for ( int j = 0; j < DIM_Z; j++ ) {
                temp += s_XZ_A[ltz][lty][j] *
                    s_ZZ_A[ltz][ltx][j];
            }
            s_XZ_K[ltz][lty][ltx] = temp;
        }

        __syncthreads();

        //  Compute self.x : self.x + cp.dot(self.K, self.y) --> X1
        temp = 0.0;
        if ( ltx == 0 ) {
#pragma unroll DIM_Z
            for ( int j = 0; j < DIM_Z; j++ ) {
                temp += s_XZ_K[ltz][lty][j] *
                s_Z1_y[ltz][j][ltx];
            }
            x_in[gtz * DIM_X * 1 + lty * 1 + ltx] += temp;
        }

        // Compute I_KH = self_I - dot(self.K, self.H) --> XX
        temp = 0.0;
#pragma unroll DIM_Z
        for ( int j = 0; j < DIM_Z; j++ ) {
            temp += s_XZ_K[ltz][lty][j] *
                s_ZX_H[ltz][j][ltx];
        }
        // s_XX_A holds I_KH
        s_XX_A[ltz][lty][ltx] = ( ( ltx == lty ) ? 1 : 0 ) - temp;

        __syncthreads();

        // Compute self.P = dot(dot(I_KH, self.P), I_KH.T) +
        // dot(dot(self.K, self.R), self.K.T)

        // Compute dot(I_KH, self.P) --> XX
        temp = 0.0;
#pragma unroll DIM_X
        for ( int j = 0; j < DIM_X; j++ ) {
            temp += s_XX_A[ltz][lty][j] *
                s_XX_P[ltz][j][ltx];
        }
        s_XX_B[ltz][lty][ltx] = temp;

        __syncthreads();

        // Compute dot(dot(I_KH, self.P), I_KH.T) --> XX
        temp = 0.0;
#pragma unroll DIM_X
        for ( int j = 0; j < DIM_X; j++ ) {
            temp += s_XX_B[ltz][lty][j] *
                s_XX_A[ltz][ltx][j];
        }

        s_XX_P[ltz][lty][ltx] = temp;

        // Compute dot(self.K, self.R) --> XZ
        temp = 0.0;
        if ( ltx < DIM_Z ) {
#pragma unroll DIM_Z
            for ( int j = 0; j < DIM_Z; j++ ) {
                temp += s_XZ_K[ltz][lty][j] *
                    s_ZZ_R[ltz][j][ltx];
            }

            // s_XZ_A holds dot(self.K, self.R)
            s_XZ_A[ltz][lty][ltx] = temp;
        }

        __syncthreads();

        // Compute dot(dot(self.K, self.R), self.K.T) --> XX
        temp = 0.0;
#pragma unroll DIM_Z
        for ( int j = 0; j < DIM_Z; j++ ) {
            temp += s_XZ_A[ltz][lty][j] *
                s_XZ_K[ltz][ltx][j];
        }

        P[gtz * DIM_X * DIM_X + x_value] =
            s_XX_P[ltz][lty][ltx] + temp;
    }
}
"""


class _cupy_predict_wrapper(object):
    def __init__(self, grid, block, stream, kernel):
        if isinstance(grid, int):
            grid = (grid,)
        if isinstance(block, int):
            block = (block,)

        self.grid = grid
        self.block = block
        self.stream = stream
        self.kernel = kernel

    def __call__(
        self, alpha_sq, x, F, P, Q,
    ):

        kernel_args = (x.shape[0], alpha_sq, x, F, P, Q)

        with self.stream:
            self.kernel(self.grid, self.block, kernel_args)


class _cupy_update_wrapper(object):
    def __init__(self, grid, block, stream, kernel):
        if isinstance(grid, int):
            grid = (grid,)
        if isinstance(block, int):
            block = (block,)

        self.grid = grid
        self.block = block
        self.stream = stream
        self.kernel = kernel

    def __call__(self, x, z, H, P, R):

        kernel_args = (x.shape[0], x, z, H, P, R)

        with self.stream:
            self.kernel(self.grid, self.block, kernel_args)


def _get_backend_kernel(dtype, grid, block, stream, use_numba, k_type):

    if not use_numba:
        kernel = _cupy_kernel_cache[(dtype.name, k_type)]
        if kernel:
            if k_type == GPUKernel.PREDICT:
                return _cupy_predict_wrapper(grid, block, stream, kernel)
            elif k_type == GPUKernel.UPDATE:
                return _cupy_update_wrapper(grid, block, stream, kernel)
            else:
                raise NotImplementedError(
                    "No CuPY kernel found for k_type {}, datatype {}".format(
                        k_type, dtype
                    )
                )
        else:
            raise ValueError(
                "Kernel {} not found in _cupy_kernel_cache".format(k_type)
            )
    else:
        nb_stream = stream_cupy_to_numba(stream)
        kernel = _numba_kernel_cache[(dtype.name, k_type)]

        if kernel:
            return kernel[grid, block, nb_stream]
        else:
            raise ValueError(
                "Kernel {} not found in _numba_kernel_cache".format(k_type)
            )
    raise NotImplementedError(
        "No kernel found for k_type {}, datatype {}".format(k_type, dtype.name)
    )


def _populate_kernel_cache(
    np_type, use_numba, dim_x, dim_z, max_tpb, min_bpsm
):

    # Check in np_type is a supported option
    try:
        numba_type, c_type = _SUPPORTED_TYPES[np_type]

    except ValueError:
        raise Exception("No kernel found for datatype {}".format(np_type))

    if not use_numba:
        # Instantiate the cupy kernel for this type and compile
        specializations = (
            "_cupy_predict<{}, {}, {}, {}>".format(
                c_type, dim_x, max_tpb, min_bpsm
            ),
            "_cupy_update<{}, {}, {}, {}, {}>".format(
                c_type, dim_x, dim_z, max_tpb, min_bpsm
            ),
        )
        module = cp.RawModule(
            code=cuda_code,
            options=("-std=c++11", "-fmad=true",),
            name_expressions=specializations,
        )

        _cupy_kernel_cache[
            (str(numba_type), GPUKernel.PREDICT)
        ] = module.get_function(specializations[0])
        _cupy_kernel_cache[
            (str(numba_type), GPUKernel.UPDATE)
        ] = module.get_function(specializations[1])
    else:
        sig = _numba_kalman_signature(numba_type)
        _numba_kernel_cache[(str(numba_type), GPUKernel.PREDICT)] = cuda.jit(
            sig
        )(_numba_predict)
        _numba_kernel_cache[(str(numba_type), GPUKernel.UPDATE)] = cuda.jit(
            sig
        )(_numba_update)
