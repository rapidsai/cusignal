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
from ..utils.helper_tools import _print_atts, _get_function


_SUPPORTED_TYPES = ["float32", "float64", "complex64", "complex128"]


class _cupy_channelizer_wrapper(object):
    def __init__(self, grid, block, kernel):
        if isinstance(grid, int):
            grid = (grid,)
        if isinstance(block, int):
            block = (block,)

        self.grid = grid
        self.block = block
        self.kernel = kernel

    def __call__(self, n_chans, n_taps, n_pts):

        kernel_args = (n_chans, n_taps, n_pts)

        self.kernel(self.grid, self.block, kernel_args)


def _populate_kernel_cache(np_type, k_type):

    if np_type not in _SUPPORTED_TYPES:
        raise ValueError(
            "Datatype {} not found for '{}'".format(np_type, k_type)
        )

    if (str(np_type), k_type) in _cupy_kernel_cache:
        return

    _cupy_kernel_cache[(str(np_type), k_type)] = _get_function(
        "/filtering/_channelizer.fatbin", "_cupy_channelizer_" + str(np_type),
    )


def _get_backend_kernel(dtype, grid, block, smem, k_type):
    kernel = _cupy_kernel_cache[(dtype.name, k_type)]
    if kernel:
        return _cupy_channelizer_wrapper(grid, block, smem, kernel)
    else:
        raise ValueError(
            "Kernel {} not found in _cupy_kernel_cache".format(k_type)
        )

    raise NotImplementedError(
        "No kernel found for datatype {}".format(dtype.name)
    )


def _channelizer(x, h, n_chans, order="C"):

    # number of taps in each h_n filter
    n_taps = int(len(h) / n_chans)

    # number of outputs
    n_pts = int(len(x) / n_chans)

    k_type = "channelizer"

    threadsperblock = (8, 8)
    blockspergrid = n_pts

    _populate_kernel_cache(x.dtype, k_type)

    kernel = _get_backend_kernel(
        x.dtype, blockspergrid, threadsperblock, k_type,
    )

    kernel(n_taps, n_pts)

    _print_atts(kernel)
