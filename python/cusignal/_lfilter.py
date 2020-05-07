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

from .utils._caches import _cupy_kernel_cache


class _cupy_lfilter_wrapper(object):
    def __init__(self, grid, block, stream, kernel):
        if isinstance(grid, int):
            grid = (grid,)
        if isinstance(block, int):
            block = (block,)

        self.grid = grid
        self.block = block
        self.stream = stream
        self.kernel = kernel

    def __call__(self, b, a, x, out):

        kernel_args = (
            x.shape[0],
            a.shape[0],
            x,
            a,
            b,
            out,
        )

        self.stream.use()
        self.kernel(self.grid, self.block, kernel_args)


def _get_backend_kernel(
    dtype, grid, block, stream, use_numba, k_type,
):
    from .utils._compile_kernels import GPUKernel
    if not use_numba:
        kernel = _cupy_kernel_cache[(dtype.name, k_type.value)]
        if kernel:
            if k_type == GPUKernel.LFILTER:
                return _cupy_lfilter_wrapper(grid, block, stream, kernel)
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

    raise NotImplementedError(
        "No kernel found for k_type {}, datatype {}".format(k_type, dtype.name)
    )


def _lfilter_gpu(b, a, x, clamp, cp_stream, autosync):
    from .utils._compile_kernels import _populate_kernel_cache, GPUKernel

    out = cp.zeros_like(x)

    threadsperblock = 1
    blockspergrid = 1

    _populate_kernel_cache(out.dtype.type, False, GPUKernel.LFILTER)
    kernel = _get_backend_kernel(
        out.dtype,
        blockspergrid,
        threadsperblock,
        cp_stream,
        False,
        GPUKernel.LFILTER,
    )

    kernel(b, a, x, out)

    if autosync is True:
        cp_stream.synchronize()

    return out
