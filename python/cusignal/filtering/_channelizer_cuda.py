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
from ..utils.helper_tools import _print_atts, _get_function, _get_numSM


_SUPPORTED_TYPES = [
    "float32_complex64",
    "complex64_complex64",
    "float64_complex128",
    "complex128_complex128",
]


class _cupy_channelizer_wrapper(object):
    def __init__(self, grid, block, kernel):
        if isinstance(grid, int):
            grid = (grid,)
        if isinstance(block, int):
            block = (block,)

        self.grid = grid
        self.block = block
        self.kernel = kernel

    def __call__(self, n_chans, n_taps, n_pts, x, h, y):

        kernel_args = (n_chans, n_taps, n_pts, x, h, y)

        self.kernel(self.grid, self.block, kernel_args)


def _populate_kernel_cache(np_type, k_type):

    if np_type not in _SUPPORTED_TYPES:
        raise ValueError(
            "Datatype {} not found for '{}'".format(np_type, k_type)
        )

    if (str(np_type), k_type) in _cupy_kernel_cache:
        return

    _cupy_kernel_cache[(str(np_type), k_type)] = _get_function(
        "/filtering/_channelizer.fatbin",
        "_cupy_" + str(k_type) + "_" + str(np_type),
    )


def _get_backend_kernel(dtype, grid, block, k_type):

    kernel = _cupy_kernel_cache[(dtype, k_type)]
    if kernel:
        return _cupy_channelizer_wrapper(grid, block, kernel)
    else:
        raise ValueError(
            "Kernel {} not found in _cupy_kernel_cache".format(k_type)
        )

    raise NotImplementedError("No kernel found for datatype {}".format(dtype))


def _channelizer(x, h, y, n_chans, n_taps, n_pts):

    np_type = str(x.dtype) + "_" + str(y.dtype)

    if n_chans <= 8 and n_taps <= 8:

        k_type = "channelizer_8x8"

        threadsperblock = (8, 8)
        blockspergrid = _get_numSM() * 32

        _populate_kernel_cache(np_type, k_type)

        kernel = _get_backend_kernel(
            np_type, blockspergrid, threadsperblock, k_type,
        )

    elif n_chans <= 16 and n_taps <= 16:

        k_type = "channelizer_16x16"

        threadsperblock = (16, 16)
        blockspergrid = _get_numSM() * 32

        _populate_kernel_cache(np_type, k_type)

        kernel = _get_backend_kernel(
            np_type, blockspergrid, threadsperblock, k_type,
        )

    elif n_chans <= 32 and n_taps <= 32:

        k_type = "channelizer_32x32"

        threadsperblock = (32, 32)
        blockspergrid = _get_numSM() * 32

        _populate_kernel_cache(np_type, k_type)

        kernel = _get_backend_kernel(
            np_type, blockspergrid, threadsperblock, k_type,
        )

    else:

        k_type = "channelizer"

        threadsperblock = (8, 8)
        blockspergrid = _get_numSM() * 32

        _populate_kernel_cache(np_type, k_type)

        kernel = _get_backend_kernel(
            np_type, blockspergrid, threadsperblock, k_type,
        )

    kernel(n_chans, n_taps, n_pts, x, h, y)

    _print_atts(kernel)
