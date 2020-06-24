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


# Custom Cupy raw kernel implementing binary writers
# Matthew Nicely - mnicely@nvidia.com
_cupy_pack_src = Template(
    """
${header}

extern "C" {

    __global__ void _cupy_pack(
        const size_t N,
        ${datatype} * __restrict__ input,
        unsigned char * __restrict__ output) {

         const int tx {
            static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) };
        const int stride { static_cast<int>(blockDim.x * gridDim.x) };

        for ( int tid = tx; tid < N; tid += stride ) {
            output[tid] = reinterpret_cast<unsigned char*>(input)[tid];
        }
    }
}
"""
)


class _cupy_pack_wrapper(object):
    def __init__(self, grid, block, kernel):
        if isinstance(grid, int):
            grid = (grid,)
        if isinstance(block, int):
            block = (block,)

        self.grid = grid
        self.block = block
        self.kernel = kernel

    def __call__(self, out_size, binary, out):

        kernel_args = (out_size, binary, out)

        self.kernel(self.grid, self.block, kernel_args)


def _get_backend_kernel(
    dtype, grid, block, k_type,
):
    from ..utils.compile_kernels import GPUKernel

    kernel = _cupy_kernel_cache[(str(dtype), k_type.value)]
    if kernel:
        if k_type == GPUKernel.PACK:
            return _cupy_pack_wrapper(grid, block, kernel)

        raise ValueError(
            "Kernel {} not found in _cupy_kernel_cache".format(k_type)
        )


def _pack(binary):

    from ..utils.compile_kernels import _populate_kernel_cache, GPUKernel

    data_size = binary.dtype.itemsize * binary.shape[0]
    out_size = data_size

    out = cp.empty_like(binary, dtype=cp.ubyte, shape=out_size)

    device_id = cp.cuda.Device()
    numSM = device_id.attributes["MultiProcessorCount"]
    blockspergrid = numSM * 20
    threadsperblock = 512

    _populate_kernel_cache(out.dtype, GPUKernel.PACK)
    kernel = _get_backend_kernel(
        out.dtype, blockspergrid, threadsperblock, GPUKernel.PACK,
    )

    kernel(out_size, binary, out)

    # Remove binary data
    del binary

    return out
