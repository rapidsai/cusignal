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


# Custom Cupy raw kernel implementing binary readers
# Matthew Nicely - mnicely@nvidia.com
_cupy_parse_sigmf_src = Template(
    """
$header

extern "C" {
    __global__ void _cupy_parse_sigmf(
        const size_t N,
        const int data_size,
        unsigned int * __restrict__ input,
        ${datatype} * __restrict__ output) {

         const int tx {
            static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) };
        const int stride { static_cast<int>(blockDim.x * gridDim.x) };

        for ( int tid = tx; tid < N; tid += stride ) {
            output[tid] = reinterpret_cast<${datatype}*>(input)[tid];
        }
    }
}
"""
)


class _cupy_parser_wrapper(object):
    def __init__(self, grid, block, stream, kernel):
        if isinstance(grid, int):
            grid = (grid,)
        if isinstance(block, int):
            block = (block,)

        self.grid = grid
        self.block = block
        self.stream = stream
        self.kernel = kernel

    def __call__(self, out_size, data_size, in1, out):

        kernel_args = (out_size, data_size, in1, out)

        with self.stream:
            self.kernel(self.grid, self.block, kernel_args)


def _get_backend_kernel(
    dtype, grid, block, stream, k_type,
):
    from ..utils.compile_kernels import GPUKernel

    kernel = _cupy_kernel_cache[(str(dtype), k_type.value)]
    if kernel:
        if k_type == GPUKernel.PARSER_SIGMF:
            return _cupy_parser_wrapper(grid, block, stream, kernel)

        raise ValueError(
            "Kernel {} not found in _cupy_kernel_cache".format(k_type)
        )


def _parser(in1, format, keep, dtype, cp_stream, autosync):

    from ..utils.compile_kernels import _populate_kernel_cache, GPUKernel

    if dtype == cp.complex64:
        data_size = 8 // in1.dtype.itemsize  # FIX Write generic

    out_size = in1.shape[0] // data_size

    out = cp.empty_like(in1, dtype=dtype, shape=out_size)

    device_id = cp.cuda.Device()
    numSM = device_id.attributes["MultiProcessorCount"]
    blockspergrid = numSM * 20
    threadsperblock = 512

    _populate_kernel_cache(out.dtype, GPUKernel.PARSER_SIGMF)
    kernel = _get_backend_kernel(
        out.dtype,
        blockspergrid,
        threadsperblock,
        cp_stream,
        GPUKernel.PARSER_SIGMF,
    )

    kernel(out_size, data_size, in1, out)

    # Remove binary data
    if keep is False:
        del in1

    if autosync is True:
        cp_stream.synchronize()

    return out
