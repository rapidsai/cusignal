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
from ..utils.helper_tools import _get_function, _print_atts

_SUPPORTED_TYPES = ["float32", "float64"]


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
            sos,
            zi,
            x,
        )

        self.kernel(self.grid, self.block, kernel_args, shared_mem=self.smem)


def _populate_kernel_cache(np_type, k_type):

    if np_type not in _SUPPORTED_TYPES:
        raise ValueError("Datatype {} not found for '{}'".format(np_type, k_type))

    if (str(np_type), k_type) in _cupy_kernel_cache:
        return

    _cupy_kernel_cache[(str(np_type), k_type)] = _get_function(
        "/filtering/_sosfilt.fatbin",
        "_cupy_" + k_type + "_" + str(np_type),
    )


def _get_backend_kernel(dtype, grid, block, smem, k_type):
    kernel = _cupy_kernel_cache[(dtype.name, k_type)]
    if kernel:
        return _cupy_sosfilt_wrapper(grid, block, smem, kernel)
    else:
        raise ValueError("Kernel {} not found in _cupy_kernel_cache".format(k_type))

    raise NotImplementedError("No kernel found for datatype {}".format(dtype.name))


def _sosfilt(sos, x, zi):

    threadsperblock = sos.shape[0]  # Up-to (1024, 1) = 1024 max per block
    blockspergrid = x.shape[0]

    k_type = "sosfilt"

    _populate_kernel_cache(x.dtype, k_type)

    out_size = threadsperblock
    sos_size = sos.shape[0] * sos.shape[1]

    shared_mem = (out_size + sos_size) * x.dtype.itemsize

    kernel = _get_backend_kernel(
        x.dtype,
        blockspergrid,
        threadsperblock,
        shared_mem,
        k_type,
    )
    print(zi.shape)

    kernel(sos, x, zi)

    _print_atts(kernel)
