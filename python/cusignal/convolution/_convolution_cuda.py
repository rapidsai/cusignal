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
import numpy as np

from ..utils._caches import _cupy_kernel_cache
from ..utils.helper_tools import _print_atts, _get_function, _get_tpb_bpg
from .convolution_utils import (
    FULL,
    SAME,
    VALID,
    CIRCULAR,
    REFLECT,
    PAD,
    _iDivUp,
    _valfrommode,
    _bvalfromboundary,
)


_SUPPORTED_TYPES = [
    "int32",
    "int64",
    "float32",
    "float64",
    "complex64",
    "complex128",
]


class _cupy_convolve_wrapper(object):
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
        d_inp,
        d_kernel,
        mode,
        swapped_inputs,
        out,
    ):

        kernel_args = (
            d_inp,
            d_inp.shape[0],
            d_kernel,
            d_kernel.shape[0],
            mode,
            swapped_inputs,
            out,
            out.shape[0],
        )

        self.kernel(self.grid, self.block, kernel_args)


class _cupy_convolve_1d2o_wrapper(object):
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
        d_inp,
        d_kernel,
        mode,
        out,
    ):

        kernel_args = (
            d_inp,
            d_inp.shape[0],
            d_kernel,
            d_kernel.shape[0],
            d_kernel.shape[1],
            mode,
            out,
            out.shape[0],
        )

        self.kernel(self.grid, self.block, kernel_args)


class _cupy_convolve_1d3o_wrapper(object):
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
        d_inp,
        d_kernel,
        mode,
        out,
    ):

        kernel_args = (
            d_inp,
            d_inp.shape[0],
            d_kernel,
            d_kernel.shape[0],
            d_kernel.shape[1],
            d_kernel.shape[2],
            mode,
            out,
            out.shape[0],
        )

        self.kernel(self.grid, self.block, kernel_args)


class _cupy_convolve_2d_wrapper(object):
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
        d_inp,
        paddedW,
        paddedH,
        d_kernel,
        S0,
        S1,
        out,
        outW,
        outH,
        pick,
    ):

        kernel_args = (
            d_inp,
            paddedW,
            paddedH,
            d_kernel,
            d_kernel.shape[0],
            d_kernel.shape[1],
            S0,
            S1,
            out,
            outW,
            outH,
            pick,
        )

        self.kernel(self.grid, self.block, kernel_args)


def _populate_kernel_cache(np_type, k_type):

    if np_type not in _SUPPORTED_TYPES:
        raise ValueError(
            "Datatype {} not found for '{}'".format(np_type, k_type)
        )

    if (str(np_type), k_type) in _cupy_kernel_cache:
        return

    _cupy_kernel_cache[(str(np_type), k_type)] = _get_function(
        "/convolution/_convolution.fatbin",
        "_cupy_" + k_type + "_" + str(np_type),
    )


def _get_backend_kernel(
    dtype,
    grid,
    block,
    k_type,
):

    kernel = _cupy_kernel_cache[(str(dtype), k_type)]
    if kernel:
        if k_type == "convolve" or k_type == "correlate":
            return _cupy_convolve_wrapper(grid, block, kernel)
        elif k_type == "convolve2D" or k_type == "correlate2D":
            return _cupy_convolve_2d_wrapper(grid, block, kernel)
        elif k_type == "convolve1D2O":
            return _cupy_convolve_1d2o_wrapper(grid, block, kernel)
        elif k_type == "convolve1D3O":
            return _cupy_convolve_1d3o_wrapper(grid, block, kernel)
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


def _convolve_gpu(
    inp,
    out,
    ker,
    mode,
    use_convolve,
    swapped_inputs,
):

    d_inp = cp.asarray(inp)
    d_kernel = cp.asarray(ker)

    threadsperblock, blockspergrid = _get_tpb_bpg()

    if use_convolve:
        k_type = "convolve"

        _populate_kernel_cache(out.dtype, k_type)

        kernel = _get_backend_kernel(
            out.dtype,
            blockspergrid,
            threadsperblock,
            k_type,
        )
    else:
        k_type = "correlate"

        _populate_kernel_cache(out.dtype, k_type)

        kernel = _get_backend_kernel(
            out.dtype,
            blockspergrid,
            threadsperblock,
            k_type,
        )

    kernel(d_inp, d_kernel, mode, swapped_inputs, out)

    _print_atts(kernel)

    return out


def _convolve2d_gpu(
    inp,
    out,
    ker,
    mode,
    boundary,
    use_convolve,
    fillvalue,
):

    if (boundary != PAD) and (boundary != REFLECT) and (boundary != CIRCULAR):
        raise Exception("Invalid boundary flag")

    S = np.zeros(2, dtype=int)

    # If kernel is square and odd
    if ker.shape[0] == ker.shape[1]:  # square
        if ker.shape[0] % 2 == 1:  # odd
            pick = 1
            S[0] = (ker.shape[0] - 1) // 2
            if mode == 2:  # full
                P1 = P2 = P3 = P4 = S[0] * 2
            else:  # same/valid
                P1 = P2 = P3 = P4 = S[0]
        else:  # even
            pick = 2
            S[0] = ker.shape[0] // 2
            if mode == 2:  # full
                P1 = P2 = P3 = P4 = S[0] * 2 - 1
            else:  # same/valid
                if use_convolve:
                    P1 = P2 = P3 = P4 = S[0]
                else:
                    P1 = P3 = S[0] - 1
                    P2 = P4 = S[0]
    else:  # Non-square
        pick = 3
        S[0] = ker.shape[0]
        S[1] = ker.shape[1]
        if mode == 2:  # full
            P1 = S[0] - 1
            P2 = S[0] - 1
            P3 = S[1] - 1
            P4 = S[1] - 1
        else:  # same/valid
            if use_convolve:
                P1 = S[0] // 2
                P2 = S[0] // 2 if (S[0] % 2) else S[0] // 2 - 1
                P3 = S[1] // 2
                P4 = S[1] // 2 if (S[1] % 2) else S[1] // 2 - 1
            else:
                P1 = S[0] // 2 if (S[0] % 2) else S[0] // 2 - 1
                P2 = S[0] // 2
                P3 = S[1] // 2 if (S[1] % 2) else S[1] // 2 - 1
                P4 = S[1] // 2

    if mode == 1:  # SAME
        pad = ((P1, P2), (P3, P4))  # 4x5
        if boundary == REFLECT:
            inp = cp.pad(inp, pad, "symmetric")
        if boundary == CIRCULAR:
            inp = cp.pad(inp, pad, "wrap")
        if boundary == PAD:
            inp = cp.pad(inp, pad, "constant", constant_values=(fillvalue))

    if mode == 2:  # FULL
        pad = ((P1, P2), (P3, P4))
        if boundary == REFLECT:
            inp = cp.pad(inp, pad, "symmetric")
        if boundary == CIRCULAR:
            inp = cp.pad(inp, pad, "wrap")
        if boundary == PAD:
            inp = cp.pad(inp, pad, "constant", constant_values=(fillvalue))

    paddedW = inp.shape[1]
    paddedH = inp.shape[0]

    outW = out.shape[1]
    outH = out.shape[0]

    d_inp = cp.asarray(inp)
    d_kernel = cp.asarray(ker)

    threadsperblock = (16, 16)
    blockspergrid = (
        _iDivUp(out.shape[1], threadsperblock[0]),
        _iDivUp(out.shape[0], threadsperblock[1]),
    )

    if use_convolve:
        k_type = "convolve2D"

        _populate_kernel_cache(out.dtype, k_type)

        kernel = _get_backend_kernel(
            out.dtype,
            blockspergrid,
            threadsperblock,
            k_type,
        )
    else:
        k_type = "correlate2D"

        _populate_kernel_cache(out.dtype, k_type)

        kernel = _get_backend_kernel(
            out.dtype,
            blockspergrid,
            threadsperblock,
            k_type,
        )

    kernel(
        d_inp, paddedW, paddedH, d_kernel, S[0], S[1], out, outW, outH, pick
    )

    _print_atts(kernel)

    return out


def _convolve(
    in1,
    in2,
    use_convolve,
    swapped_inputs,
    mode,
):

    val = _valfrommode(mode)

    # Promote inputs
    promType = cp.promote_types(in1.dtype, in2.dtype)
    in1 = in1.astype(promType)
    in2 = in2.astype(promType)

    # Create empty array to hold number of aout dimensions
    out_dimens = np.empty(in1.ndim, np.int)
    if val == VALID:
        for i in range(in1.ndim):
            out_dimens[i] = (
                max(in1.shape[i], in2.shape[i])
                - min(in1.shape[i], in2.shape[i])
                + 1
            )
            if out_dimens[i] < 0:
                raise Exception(
                    "no part of the output is valid, use option 1 (same) or 2 \
                     (full) for third argument"
                )
    elif val == SAME:
        for i in range(in1.ndim):
            if not swapped_inputs:
                out_dimens[i] = in1.shape[i]  # Per scipy docs
            else:
                out_dimens[i] = min(in1.shape[i], in2.shape[i])
    elif val == FULL:
        for i in range(in1.ndim):
            out_dimens[i] = in1.shape[i] + in2.shape[i] - 1
    else:
        raise Exception("mode must be 0 (valid), 1 (same), or 2 (full)")

    # Create empty array out on GPU
    out = cp.empty(out_dimens.tolist(), in1.dtype)

    out = _convolve_gpu(
        in1,
        out,
        in2,
        val,
        use_convolve,
        swapped_inputs,
    )

    return out


def _convolve2d(in1, in2, use_convolve, mode, boundary, fillvalue):

    val = _valfrommode(mode)
    bval = _bvalfromboundary(boundary)

    # Promote inputs
    promType = cp.promote_types(in1.dtype, in2.dtype)
    in1 = in1.astype(promType)
    in2 = in2.astype(promType)

    if (bval != PAD) and (bval != REFLECT) and (bval != CIRCULAR):
        raise Exception("Incorrect boundary value.")

    if (bval == PAD) and (fillvalue is not None):
        fill = np.array(fillvalue, in1.dtype)
        if fill is None:
            raise Exception("fill must no be None.")
        if fill.size != 1:
            if fill.size == 0:
                raise Exception("`fillvalue` cannot be an empty array.")
            raise Exception(
                "`fillvalue` must be scalar or an array with one element"
            )
    else:
        fill = np.zeros(1, in1.dtype)
        if fill is None:
            raise Exception("Unable to create fill array")

    # Create empty array to hold number of aout dimensions
    out_dimens = np.empty(in1.ndim, np.int)
    if val == VALID:
        for i in range(in1.ndim):
            out_dimens[i] = in1.shape[i] - in2.shape[i] + 1
            if out_dimens[i] < 0:
                raise Exception(
                    "no part of the output is valid, use option 1 (same) or 2 \
                     (full) for third argument"
                )
    elif val == SAME:
        for i in range(in1.ndim):
            out_dimens[i] = in1.shape[i]
    elif val == FULL:
        for i in range(in1.ndim):
            out_dimens[i] = in1.shape[i] + in2.shape[i] - 1
    else:
        raise Exception("mode must be 0 (valid), 1 (same), or 2 (full)")

    # Create empty array out on GPU
    out = cp.empty(out_dimens.tolist(), in1.dtype)

    out = _convolve2d_gpu(
        in1,
        out,
        in2,
        val,
        bval,
        use_convolve,
        fill,
    )

    return out


def _convolve1d2o_gpu(
    inp,
    out,
    ker,
    mode,
):

    d_inp = cp.asarray(inp)
    d_kernel = cp.asarray(ker)

    threadsperblock, blockspergrid = _get_tpb_bpg()

    k_type = "convolve1D2O"

    _populate_kernel_cache(out.dtype, k_type)

    kernel = _get_backend_kernel(
        out.dtype,
        blockspergrid,
        threadsperblock,
        k_type,
    )

    kernel(d_inp, d_kernel, mode, out)

    _print_atts(kernel)

    return out


def _convolve1d2o(in1, in2, mode):

    val = _valfrommode(mode)

    # Promote inputs
    promType = cp.promote_types(in1.dtype, in2.dtype)
    in1 = in1.astype(promType)
    in2 = in2.astype(promType)

    # Create empty array to hold number of aout dimensions
    out_dimens = np.empty(in1.ndim, np.int)
    if val == VALID:
        for i in range(in1.ndim):
            out_dimens[i] = in1.shape[i] - in2.shape[i] + 1
            if out_dimens[i] < 0:
                raise Exception(
                    "no part of the output is valid, use option 1 (same) or 2 \
                     (full) for third argument"
                )

    # Create empty array out on GPU
    out = cp.empty(out_dimens.tolist(), in1.dtype)

    out = _convolve1d2o_gpu(
        in1,
        out,
        in2,
        val,
    )

    return out


def _convolve1d3o_gpu(
    inp,
    out,
    ker,
    mode,
):

    d_inp = cp.asarray(inp)
    d_kernel = cp.asarray(ker)

    threadsperblock, blockspergrid = _get_tpb_bpg()

    k_type = "convolve1D3O"

    _populate_kernel_cache(out.dtype, k_type)

    kernel = _get_backend_kernel(
        out.dtype,
        blockspergrid,
        threadsperblock,
        k_type,
    )

    kernel(d_inp, d_kernel, mode, out)

    _print_atts(kernel)

    return out


def _convolve1d3o(in1, in2, mode):

    val = _valfrommode(mode)

    # Promote inputs
    promType = cp.promote_types(in1.dtype, in2.dtype)
    in1 = in1.astype(promType)
    in2 = in2.astype(promType)

    # Create empty array to hold number of aout dimensions
    out_dimens = np.empty(in1.ndim, np.int)
    if val == VALID:
        for i in range(in1.ndim):
            out_dimens[i] = in1.shape[i] - in2.shape[i] + 1
            if out_dimens[i] < 0:
                raise Exception(
                    "no part of the output is valid, use option 1 (same) or 2 \
                     (full) for third argument"
                )

    # Create empty array out on GPU
    out = cp.empty(out_dimens.tolist(), in1.dtype)

    out = _convolve1d3o_gpu(
        in1,
        out,
        in2,
        val,
    )

    return out
