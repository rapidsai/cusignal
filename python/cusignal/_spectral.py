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
import warnings

from ._caches import _cupy_kernel_cache, _numba_kernel_cache
from ._compile_kernels import (
    _stream_cupy_to_numba,
    _populate_kernel_cache,
    GPUKernel,
)

# Display FutureWarnings only once per module
warnings.simplefilter("once", FutureWarning)


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


def _get_backend_kernel(dtype, grid, block, stream, use_numba, k_type):

    if not use_numba:
        kernel = _cupy_kernel_cache[(dtype.name, k_type.value)]
        if kernel:
            return _cupy_lombscargle_wrapper(grid, block, stream, kernel)
        else:
            raise ValueError(
                "Kernel {} not found in _cupy_kernel_cache".format(k_type)
            )

    else:
        warnings.warn(
            "Numba kernels will be removed in a later release",
            FutureWarning,
            stacklevel=4,
        )

        nb_stream = _stream_cupy_to_numba(stream)
        kernel = _numba_kernel_cache[(dtype.name, k_type.value)]

        if kernel:
            return kernel[grid, block, nb_stream]
        else:
            raise ValueError(
                "Kernel {} not found in _numba_kernel_cache".format(k_type)
            )

    raise NotImplementedError(
        "No kernel found for datatype {}".format(dtype.name)
    )


def _lombscargle(x, y, freqs, pgram, y_dot, cp_stream, autosync, use_numba):

    device_id = cp.cuda.Device()
    numSM = device_id.attributes["MultiProcessorCount"]
    threadsperblock = 256
    blockspergrid = numSM * 20

    _populate_kernel_cache(pgram.dtype.type, use_numba, GPUKernel.LOMBSCARGLE)

    kernel = _get_backend_kernel(
        pgram.dtype,
        blockspergrid,
        threadsperblock,
        cp_stream,
        use_numba,
        GPUKernel.LOMBSCARGLE,
    )

    kernel(x, y, freqs, pgram, y_dot)

    if autosync is True:
        cp_stream.synchronize()
