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

import cupy as cp

from ..convolution.convolution_utils import _iDivUp
from ..utils._caches import _cupy_kernel_cache
from ..utils.helper_tools import _get_function, _get_tpb_bpg, _print_atts

_modedict = {
    cp.less: 0,
    cp.greater: 1,
    cp.less_equal: 2,
    cp.greater_equal: 3,
    cp.equal: 4,
    cp.not_equal: 5,
}

_SUPPORTED_TYPES = [
    "int32",
    "int64",
    "float32",
    "float64",
]


class _cupy_boolrelextrema_1d_wrapper(object):
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
        data,
        comp,
        axis,
        order,
        clip,
        out,
    ):

        kernel_args = (data.shape[axis], order, clip, comp, data, out)

        self.kernel(self.grid, self.block, kernel_args)


class _cupy_boolrelextrema_2d_wrapper(object):
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
        data,
        comp,
        axis,
        order,
        clip,
        out,
    ):

        kernel_args = (
            data.shape[1],
            data.shape[0],
            order,
            clip,
            comp,
            axis,
            data,
            out,
        )

        self.kernel(self.grid, self.block, kernel_args)


def _populate_kernel_cache(np_type, k_type):

    if np_type not in _SUPPORTED_TYPES:
        raise ValueError("Datatype {} not found for '{}'".format(np_type, k_type))

    if (str(np_type), k_type) in _cupy_kernel_cache:
        return

    _cupy_kernel_cache[(str(np_type), k_type)] = _get_function(
        "/peak_finding/_peak_finding.fatbin",
        "_cupy_" + k_type + "_" + str(np_type),
    )


def _get_backend_kernel(dtype, grid, block, k_type):

    kernel = _cupy_kernel_cache[(str(dtype), k_type)]
    if kernel:
        if k_type == "boolrelextrema_1D":
            return _cupy_boolrelextrema_1d_wrapper(grid, block, kernel)
        else:
            return _cupy_boolrelextrema_2d_wrapper(grid, block, kernel)
    else:
        raise ValueError("Kernel {} not found in _cupy_kernel_cache".format(k_type))


def _peak_finding(data, comparator, axis, order, mode, results):

    comp = _modedict[comparator]

    if mode == "clip":
        clip = True
    else:
        clip = False

    if data.ndim == 1:
        k_type = "boolrelextrema_1D"

        threadsperblock, blockspergrid = _get_tpb_bpg()

        _populate_kernel_cache(data.dtype, k_type)

        kernel = _get_backend_kernel(
            data.dtype,
            blockspergrid,
            threadsperblock,
            k_type,
        )
    else:
        k_type = "boolrelextrema_2D"

        threadsperblock = (16, 16)
        blockspergrid = (
            _iDivUp(data.shape[1], threadsperblock[0]),
            _iDivUp(data.shape[0], threadsperblock[1]),
        )

        _populate_kernel_cache(data.dtype, k_type)

        kernel = _get_backend_kernel(
            data.dtype,
            blockspergrid,
            threadsperblock,
            k_type,
        )

    kernel(data, comp, axis, order, clip, results)

    _print_atts(kernel)
