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

from string import Template

from ..utils._caches import _cupy_kernel_cache


# Custom Cupy raw kernel implementing lombscargle operation
# Matthew Nicely - mnicely@nvidia.com
_cupy_lombscargle_src = Template(
    """
$header

extern "C" {
    __global__ void _cupy_lombscargle(
            const int x_shape,
            const int freqs_shape,
            const ${datatype} * __restrict__ x,
            const ${datatype} * __restrict__ y,
            const ${datatype} * __restrict__ freqs,
            ${datatype} * __restrict__ pgram,
            const ${datatype} * __restrict__ y_dot
            ) {

        const int tx {
            static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
        const int stride { static_cast<int>( blockDim.x * gridDim.x ) };

        ${datatype} yD {};
        if ( y_dot[0] == 0 ) {
            yD = 1.0f;
        } else {
            yD = 2.0f / y_dot[0];
        }

        for ( int tid = tx; tid < freqs_shape; tid += stride ) {

            ${datatype} freq { freqs[tid] };

            ${datatype} xc {};
            ${datatype} xs {};
            ${datatype} cc {};
            ${datatype} ss {};
            ${datatype} cs {};
            ${datatype} c {};
            ${datatype} s {};

            for ( int j = 0; j < x_shape; j++ ) {
                c = cos( freq * x[j] );
                s = sin( freq * x[j] );

                xc += y[j] * c;
                xs += y[j] * s;
                cc += c * c;
                ss += s * s;
                cs += c * s;
            }

            ${datatype} tau { atan2( 2.0f * cs, cc - ss ) / ( 2.0f * freq ) };
            ${datatype} c_tau { cos(freq * tau) };
            ${datatype} s_tau { sin(freq * tau) };
            ${datatype} c_tau2 { c_tau * c_tau };
            ${datatype} s_tau2 { s_tau * s_tau };
            ${datatype} cs_tau { 2.0f * c_tau * s_tau };

            pgram[tid] = (
                0.5f * (
                   (
                       ( c_tau * xc + s_tau * xs )
                       * ( c_tau * xc + s_tau * xs )
                       / ( c_tau2 * cc + cs_tau * cs + s_tau2 * ss )
                    )
                   + (
                       ( c_tau * xs - s_tau * xc )
                       * ( c_tau * xs - s_tau * xc )
                       / ( c_tau2 * ss - cs_tau * cs + s_tau2 * cc )
                    )
                )
            ) * yD;
        }
    }
}
"""
)


class _cupy_lombscargle_wrapper(object):
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
        self, x, y, freqs, pgram, y_dot,
    ):

        kernel_args = (
            x.shape[0],
            freqs.shape[0],
            x,
            y,
            freqs,
            pgram,
            y_dot,
        )

        self.stream.use()
        self.kernel(self.grid, self.block, kernel_args)


def _get_backend_kernel(dtype, grid, block, stream, k_type):

    kernel = _cupy_kernel_cache[(dtype.name, k_type.value)]
    if kernel:
        return _cupy_lombscargle_wrapper(grid, block, stream, kernel)
    else:
        raise ValueError(
            "Kernel {} not found in _cupy_kernel_cache".format(k_type)
        )


def _lombscargle(x, y, freqs, pgram, y_dot, cp_stream, autosync):
    from ..utils.compile_kernels import _populate_kernel_cache, GPUKernel

    device_id = cp.cuda.Device()
    numSM = device_id.attributes["MultiProcessorCount"]
    threadsperblock = 256
    blockspergrid = numSM * 20

    _populate_kernel_cache(pgram.dtype.type, GPUKernel.LOMBSCARGLE)

    kernel = _get_backend_kernel(
        pgram.dtype,
        blockspergrid,
        threadsperblock,
        cp_stream,
        GPUKernel.LOMBSCARGLE,
    )

    kernel(x, y, freqs, pgram, y_dot)

    if autosync is True:
        cp_stream.synchronize()
