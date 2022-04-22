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
