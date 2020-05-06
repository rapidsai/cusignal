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
import warnings

from math import gcd

from _caches import _cupy_kernel_cache, _numba_kernel_cache
from _precompile import (
    _stream_cupy_to_numba,
    # _cupy_kernel_cache,
    # _numba_kernel_cache,
    _populate_kernel_cache,
    GPUKernel,
)

from .filter_design.fir_filter_design import firwin

# Display FutureWarnings only once per module
warnings.simplefilter("once", FutureWarning)


FULL = 2
SAME = 1
VALID = 0

CIRCULAR = 8
REFLECT = 4
PAD = 0

_modedict = {"valid": 0, "same": 1, "full": 2}

_boundarydict = {
    "fill": 0,
    "pad": 0,
    "wrap": 2,
    "circular": 2,
    "symm": 1,
    "symmetric": 1,
    "reflect": 4,
}


def _valfrommode(mode):
    try:
        return _modedict[mode]
    except KeyError:
        raise ValueError(
            "Acceptable mode flags are 'valid'," " 'same', or 'full'."
        )


def _bvalfromboundary(boundary):
    try:
        return _boundarydict[boundary] << 2
    except KeyError:
        raise ValueError(
            "Acceptable boundary flags are 'fill', 'circular' "
            "(or 'wrap'), and 'symmetric' (or 'symm')."
        )


def _iDivUp(a, b):
    return (a // b + 1) if (a % b != 0) else (a // b)


class _cupy_convolve_wrapper(object):
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
        self, d_inp, d_kernel, mode, swapped_inputs, out,
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

        self.stream.use()
        self.kernel(self.grid, self.block, kernel_args)


class _cupy_convolve_2d_wrapper(object):
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
        self, d_inp, paddedW, paddedH, d_kernel, S0, S1, out, outW, outH, pick,
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

        self.stream.use()
        self.kernel(self.grid, self.block, kernel_args)


def _get_backend_kernel(
    dtype, grid, block, stream, use_numba, k_type,
):

    if not use_numba:
        kernel = _cupy_kernel_cache[(dtype.name, k_type.value)]
        if kernel:
            if k_type == GPUKernel.CONVOLVE or k_type == GPUKernel.CORRELATE:
                return _cupy_convolve_wrapper(grid, block, stream, kernel)
            elif (
                k_type == GPUKernel.CONVOLVE2D
                or k_type == GPUKernel.CORRELATE2D
            ):
                return _cupy_convolve_2d_wrapper(grid, block, stream, kernel)
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
        "No kernel found for k_type {}, datatype {}".format(k_type, dtype.name)
    )


def _convolve_gpu(
    inp, out, ker, mode, use_convolve, swapped_inputs, cp_stream,
):

    d_inp = cp.array(inp)
    d_kernel = cp.array(ker)

    device_id = cp.cuda.Device()
    numSM = device_id.attributes["MultiProcessorCount"]

    threadsperblock = 256
    blockspergrid = numSM * 20

    if use_convolve:
        _populate_kernel_cache(out.dtype.type, False, GPUKernel.CONVOLVE)
        kernel = _get_backend_kernel(
            out.dtype,
            blockspergrid,
            threadsperblock,
            cp_stream,
            False,
            GPUKernel.CONVOLVE,
        )
    else:
        _populate_kernel_cache(out.dtype.type, False, GPUKernel.CORRELATE)
        kernel = _get_backend_kernel(
            out.dtype,
            blockspergrid,
            threadsperblock,
            cp_stream,
            False,
            GPUKernel.CORRELATE,
        )

    kernel(d_inp, d_kernel, mode, swapped_inputs, out)

    return out


def _convolve2d_gpu(
    inp,
    out,
    ker,
    mode,
    boundary,
    use_convolve,
    fillvalue,
    cp_stream,
    use_numba,
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

    d_inp = cp.array(inp)
    d_kernel = cp.array(ker)

    threadsperblock = (16, 16)
    blockspergrid = (
        _iDivUp(outW, threadsperblock[0]),
        _iDivUp(outH, threadsperblock[1]),
    )

    if use_convolve:
        _populate_kernel_cache(out.dtype.type, use_numba, GPUKernel.CONVOLVE2D)
        kernel = _get_backend_kernel(
            out.dtype,
            blockspergrid,
            threadsperblock,
            cp_stream,
            use_numba,
            GPUKernel.CONVOLVE2D,
        )
    else:
        _populate_kernel_cache(
            out.dtype.type, use_numba, GPUKernel.CORRELATE2D
        )
        kernel = _get_backend_kernel(
            out.dtype,
            blockspergrid,
            threadsperblock,
            cp_stream,
            use_numba,
            GPUKernel.CORRELATE2D,
        )

    kernel(
        d_inp, paddedW, paddedH, d_kernel, S[0], S[1], out, outW, outH, pick
    )
    return out


def _convolve(
    in1,
    in2,
    use_convolve,
    swapped_inputs,
    mode,
    cp_stream=cp.cuda.stream.Stream(null=True),
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
        in1, out, in2, val, use_convolve, swapped_inputs, cp_stream=cp_stream,
    )

    return out


def _convolve2d(
    in1,
    in2,
    use_convolve,
    mode="full",
    boundary="fill",
    fillvalue=0,
    cp_stream=cp.cuda.stream.Stream(null=True),
    use_numba=False,
):

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
        cp_stream=cp_stream,
        use_numba=use_numba,
    )

    return out


def _design_resample_poly(up, down, window):
    """
    Design a prototype FIR low-pass filter using the window method
    for use in polyphase rational resampling.

    Parameters
    ----------
    up : int
        The upsampling factor.
    down : int
        The downsampling factor.
    window : string or tuple
        Desired window to use to design the low-pass filter.
        See below for details.

    Returns
    -------
    h : array
        The computed FIR filter coefficients.

    See Also
    --------
    resample_poly : Resample up or down using the polyphase method.

    Notes
    -----
    The argument `window` specifies the FIR low-pass filter design.
    The functions `cusignal.get_window` and `cusignal.firwin`
    are called to generate the appropriate filter coefficients.

    The returned array of coefficients will always be of data type
    `complex128` to maintain precision. For use in lower-precision
    filter operations, this array should be converted to the desired
    data type before providing it to `cusignal.resample_poly`.

    """

    # Determine our up and down factors
    # Use a rational approimation to save computation time on really long
    # signals
    g_ = gcd(up, down)
    up //= g_
    down //= g_

    # Design a linear-phase low-pass FIR filter
    max_rate = max(up, down)
    f_c = 1.0 / max_rate  # cutoff of FIR filter (rel. to Nyquist)

    # reasonable cutoff for our sinc-like function
    half_len = 10 * max_rate

    h = firwin(2 * half_len + 1, f_c, window=window)
    return h
