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

from ..utils._caches import _cupy_kernel_cache
from ..utils.debugtools import print_atts


class _cupy_sosfilt_wrapper(object):
    def __init__(self, grid, block, smem, kernel):
        if isinstance(grid, int):
            grid = (grid,)
        if isinstance(block, int):
            block = (block,)

        self.grid = grid
        self.block = block
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

        self.kernel(self.grid, self.block, kernel_args, shared_mem=self.smem)


def _get_backend_kernel(dtype, grid, block, smem, k_type):
    kernel = _cupy_kernel_cache[(dtype.name, k_type.value)]
    if kernel:
        return _cupy_sosfilt_wrapper(grid, block, smem, kernel)
    else:
        raise ValueError(
            "Kernel {} not found in _cupy_kernel_cache".format(k_type)
        )

    raise NotImplementedError(
        "No kernel found for datatype {}".format(dtype.name)
    )


def _sosfilt(sos, x, zi):
    from ..utils.compile_kernels import _populate_kernel_cache, GPUKernel

    threadsperblock = (sos.shape[0], 1)  # Up-to (1024, 1) = 1024 max per block
    blockspergrid = (1, x.shape[0])

    _populate_kernel_cache(x.dtype, GPUKernel.SOSFILT)

    out_size = threadsperblock[0]
    z_size = zi.shape[1] * zi.shape[2]
    sos_size = sos.shape[0] * sos.shape[1]

    shared_mem = (out_size + z_size + sos_size) * x.dtype.itemsize

    kernel = _get_backend_kernel(
        x.dtype, blockspergrid, threadsperblock, shared_mem, GPUKernel.SOSFILT,
    )

    kernel(sos, x, zi)

    print_atts(kernel)
