# Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from ..utils._caches import _cupy_kernel_cache
from ..utils.helper_tools import _get_function, _get_numSM, _print_atts

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
        raise ValueError("Datatype {} not found for '{}'".format(np_type, k_type))

    if (str(np_type), k_type) in _cupy_kernel_cache:
        return

    _cupy_kernel_cache[(str(np_type), k_type)] = _get_function(
        "/filtering/_channelizer.fatbin",
        "_cupy_" + k_type + "_" + str(np_type),
    )


def _get_backend_kernel(dtype, grid, block, k_type):

    kernel = _cupy_kernel_cache[(dtype, k_type)]
    if kernel:
        return _cupy_channelizer_wrapper(grid, block, kernel)
    else:
        raise ValueError("Kernel {} not found in _cupy_kernel_cache".format(k_type))

    raise NotImplementedError("No kernel found for datatype {}".format(dtype))


def _channelizer(x, h, y, n_chans, n_taps, n_pts):

    #  Blocks per grid sized for 2048 threads per SM
    np_type = str(x.dtype) + "_" + str(y.dtype)

    if n_taps <= 8:
        k_type = "channelizer_8x8"

        threadsperblock = (8, 8)
        blockspergrid = ((n_chans + 7) // 8, _get_numSM() * 32)

        _populate_kernel_cache(np_type, k_type)

        kernel = _get_backend_kernel(
            np_type,
            blockspergrid,
            threadsperblock,
            k_type,
        )

    elif n_taps <= 16:
        k_type = "channelizer_16x16"

        threadsperblock = (16, 16)
        blockspergrid = ((n_chans + 15) // 16, _get_numSM() * 8)

        _populate_kernel_cache(np_type, k_type)

        kernel = _get_backend_kernel(
            np_type,
            blockspergrid,
            threadsperblock,
            k_type,
        )

    elif n_taps <= 32:
        k_type = "channelizer_32x32"

        threadsperblock = (32, 32)
        blockspergrid = ((n_chans + 31) // 32, _get_numSM() * 2)

        _populate_kernel_cache(np_type, k_type)

        kernel = _get_backend_kernel(
            np_type,
            blockspergrid,
            threadsperblock,
            k_type,
        )

    kernel(n_chans, n_taps, n_pts, x, h, y)

    _print_atts(kernel)
