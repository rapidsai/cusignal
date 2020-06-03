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

from string import Template

from ..utils._caches import _cupy_kernel_cache


# Custom Cupy raw kernel implementing lombscargle operation
# Matthew Nicely - mnicely@nvidia.com
_cupy_sosfilt_src = Template(
    """
$header

extern "C" {
    __global__ void _cupy_sosfilt(
        const int n_signals,
        const int n_samples,
        const int n_sections,
        const int zi_width,
        const int sos_width,
        const ${datatype} * __restrict__ sos,
        const ${datatype} * __restrict__ zi,
        ${datatype} * __restrict__ x_in
     ) {

        extern __shared__ ${datatype} s_buffer[];

        ${datatype} *s_out { s_buffer };
        ${datatype} *s_zi {
            reinterpret_cast<${datatype}*>(&s_out[n_sections]) };
        ${datatype} *s_sos {
            reinterpret_cast<${datatype}*>(
                &s_zi[n_sections * zi_width]) };

        const int tx { static_cast<int>( threadIdx.x ) };
        const int ty {
            static_cast<int>( blockIdx.y * blockDim.y + threadIdx.y ) };

        // Reset shared memory
        s_out[tx] = 0;

        // Load zi
        for ( int i = 0; i < zi_width; i++ ) {
            s_zi[tx * zi_width + i] =
                zi[ty * n_sections * zi_width + tx * zi_width + i ];
        }

        // Load SOS
        // b is in s_sos[tx * sos_width + [0-2]]
        // a is in s_sos[tx * sos_width + [3-6]]
        for ( int i = 0; i < sos_width; i++ ) {
            s_sos[tx * sos_width + i] = sos[tx * sos_width + i];
        }

        __syncthreads();

        const int load_size { n_sections - 1 };
        const int unload_size { n_samples - load_size };

        ${datatype} temp {};
        ${datatype} x_n {};

        if ( ty < n_signals ) {
            // Loading phase
            for ( int n = 0; n < load_size; n++ ) {
                if ( tx == 0 ) {
                    x_n = x_in[ty * n_samples + n];
                } else {
                    x_n = s_out[tx - 1];
                }

                // Use direct II transposed structure
                temp = s_sos[tx * sos_width + 0]
                    * x_n + s_zi[tx * zi_width + 0];

                s_zi[tx * zi_width + 0] =
                    s_sos[tx * sos_width + 1] * x_n
                    - s_sos[tx * sos_width + 4] * temp
                    + s_zi[tx * zi_width + 1];

                s_zi[tx * zi_width + 1] =
                    s_sos[tx * sos_width + 2] * x_n
                    - s_sos[tx * sos_width + 5] * temp;

                s_out[tx] = temp;

                __syncthreads();
            }

            // Processing phase
            for ( int n = load_size; n < n_samples; n++ ) {
                if ( tx == 0 ) {
                    x_n = x_in[ty * n_samples + n];
                } else {
                    x_n = s_out[tx - 1];
                }

                // Use direct II transposed structure
                temp = s_sos[tx * sos_width + 0] *
                    x_n + s_zi[tx * zi_width + 0];

                s_zi[tx * zi_width + 0] =
                    s_sos[tx * sos_width + 1] * x_n
                    - s_sos[tx * sos_width + 4] * temp
                    + s_zi[tx * zi_width + 1];

                s_zi[tx * zi_width + 1] =
                    s_sos[tx * sos_width + 2] * x_n
                    - s_sos[tx * sos_width + 5] * temp;

                if ( tx < load_size ) {
                    s_out[tx] = temp;
                } else {
                    x_in[ty * n_samples + ( n - load_size )] = temp;
                }

                __syncthreads();
            }

            // Unloading phase
            for ( int n = 0; n < n_sections; n++ ) {
                // retire threads that are less than n
                if ( tx > n ) {
                    x_n = s_out[tx - 1];

                    // Use direct II transposed structure
                    temp = s_sos[tx * sos_width + 0] *
                        x_n + s_zi[tx * zi_width + 0];

                    s_zi[tx * zi_width + 0] =
                        s_sos[tx * sos_width + 1] * x_n
                        - s_sos[tx * sos_width + 4] * temp
                        + s_zi[tx * zi_width + 1];

                    s_zi[tx * zi_width + 1] =
                        s_sos[tx * sos_width + 2] * x_n
                        - s_sos[tx * sos_width + 5] * temp;

                    if ( tx < load_size ) {
                        s_out[tx] = temp;
                    } else {
                        x_in[ty * n_samples + ( n + unload_size )] = temp;
                    }
                    __syncthreads();
                }
            }
        }
    }
}
"""
)


class _cupy_sosfilt_wrapper(object):
    def __init__(self, grid, block, stream, smem, kernel):
        if isinstance(grid, int):
            grid = (grid,)
        if isinstance(block, int):
            block = (block,)

        self.grid = grid
        self.block = block
        self.stream = stream
        self.smem = smem
        self.kernel = kernel

    def __call__(self, sos, x, zi):

        kernel_args = (
            x.shape[0],
            x.shape[1],
            sos.shape[0],
            zi.shape[2],
            sos.shape[1],
            sos,
            zi,
            x,
        )

        with self.stream:
            self.kernel(self.grid, self.block, kernel_args, shared_mem=self.smem)


def _get_backend_kernel(dtype, grid, block, smem, stream, k_type):
    kernel = _cupy_kernel_cache[(dtype.name, k_type.value)]
    if kernel:
        return _cupy_sosfilt_wrapper(grid, block, stream, smem, kernel)
    else:
        raise ValueError(
            "Kernel {} not found in _cupy_kernel_cache".format(k_type)
        )

    raise NotImplementedError(
        "No kernel found for datatype {}".format(dtype.name)
    )


def _sosfilt(sos, x, zi, cp_stream, autosync):
    from ..utils.compile_kernels import _populate_kernel_cache, GPUKernel

    threadsperblock = (sos.shape[0], 1)  # Up-to (1024, 1) = 1024 max per block
    blockspergrid = (1, x.shape[0])

    _populate_kernel_cache(x.dtype.type, False, GPUKernel.SOSFILT)

    out_size = threadsperblock[0]
    z_size = zi.shape[1] * zi.shape[2]
    sos_size = sos.shape[0] * sos.shape[1]

    shared_mem = (out_size + z_size + sos_size) * x.dtype.itemsize

    kernel = _get_backend_kernel(
        x.dtype,
        blockspergrid,
        threadsperblock,
        shared_mem,
        cp_stream,
        GPUKernel.SOSFILT,
    )

    kernel(sos, x, zi)

    if autosync is True:
        cp_stream.synchronize()
