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

from ..utils._caches import _cupy_kernel_cache


class _cupy_unpack_wrapper(object):
    def __init__(self, grid, block, kernel):
        if isinstance(grid, int):
            grid = (grid,)
        if isinstance(block, int):
            block = (block,)

        self.grid = grid
        self.block = block
        self.kernel = kernel

    def __call__(self, out_size, little, binary, out):

        kernel_args = (out_size, little, binary, out)

        self.kernel(self.grid, self.block, kernel_args)


def _get_backend_kernel(
    dtype, grid, block, k_type,
):
    from ..utils.compile_kernels import GPUKernel

    kernel = _cupy_kernel_cache[(str(dtype), k_type.value)]
    if kernel:
        if k_type == GPUKernel.UNPACK:
            return _cupy_unpack_wrapper(grid, block, kernel)

        raise ValueError(
            "Kernel {} not found in _cupy_kernel_cache".format(k_type)
        )


def _unpack(binary, dtype, endianness):

    from ..utils.compile_kernels import _populate_kernel_cache, GPUKernel

    data_size = cp.dtype(dtype).itemsize // binary.dtype.itemsize

    out_size = binary.shape[0] // data_size

    out = cp.empty_like(binary, dtype=dtype, shape=out_size)

    if endianness == "B":
        little = False
    else:
        little = True

    device_id = cp.cuda.Device()
    numSM = device_id.attributes["MultiProcessorCount"]
    blockspergrid = numSM * 20
    threadsperblock = 512

    _populate_kernel_cache(out.dtype, GPUKernel.UNPACK)
    kernel = _get_backend_kernel(
        out.dtype, blockspergrid, threadsperblock, GPUKernel.UNPACK,
    )

    kernel(out_size, little, binary, out)

    # Remove binary data
    del binary

    return out
