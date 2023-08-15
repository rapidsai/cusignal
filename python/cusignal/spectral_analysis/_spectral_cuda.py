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
from ..utils.helper_tools import _get_function, _get_tpb_bpg, _print_atts

_SUPPORTED_TYPES = ["float32", "float64"]


class _cupy_lombscargle_wrapper(object):
    def __init__(self, grid, block, kernel):
        if isinstance(grid, int):
            grid = (grid,)
        if isinstance(block, int):
            block = (block,)

        self.grid = grid
        self.block = block
        self.kernel = kernel

    def __call__(
        self,
        x,
        y,
        freqs,
        pgram,
        y_dot,
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

        self.kernel(self.grid, self.block, kernel_args)


def _populate_kernel_cache(np_type, k_type):

    if np_type not in _SUPPORTED_TYPES:
        raise ValueError("Datatype {} not found for '{}'".format(np_type, k_type))

    if (str(np_type), k_type) in _cupy_kernel_cache:
        return

    _cupy_kernel_cache[(str(np_type), k_type)] = _get_function(
        "/spectral_analysis/_spectral.fatbin",
        "_cupy_" + k_type + "_" + str(np_type),
    )


def _get_backend_kernel(dtype, grid, block, k_type):

    kernel = _cupy_kernel_cache[(str(dtype), k_type)]
    if kernel:
        return _cupy_lombscargle_wrapper(grid, block, kernel)
    else:
        raise ValueError("Kernel {} not found in _cupy_kernel_cache".format(k_type))


def _lombscargle(x, y, freqs, pgram, y_dot):

    threadsperblock, blockspergrid = _get_tpb_bpg()

    k_type = "lombscargle"

    _populate_kernel_cache(pgram.dtype, k_type)

    kernel = _get_backend_kernel(
        pgram.dtype,
        blockspergrid,
        threadsperblock,
        k_type,
    )

    kernel(x, y, freqs, pgram, y_dot)

    _print_atts(kernel)
