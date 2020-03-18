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

import math
from numba import cuda

import numpy as np
import cupy as cp
from .fir_filter_design import firwin

FULL = 2
SAME = 1
VALID = 0

CIRCULAR = 8
REFLECT = 4
PAD = 0

_modedict = {'valid': 0, 'same': 1, 'full': 2}

_boundarydict = {'fill': 0, 'pad': 0, 'wrap': 2, 'circular': 2, 'symm': 1,
                 'symmetric': 1, 'reflect': 4}


def _valfrommode(mode):
    try:
        return _modedict[mode]
    except KeyError:
        raise ValueError("Acceptable mode flags are 'valid',"
                         " 'same', or 'full'.")


def _bvalfromboundary(boundary):
    try:
        return _boundarydict[boundary] << 2
    except KeyError:
        raise ValueError("Acceptable boundary flags are 'fill', 'circular' "
                         "(or 'wrap'), and 'symmetric' (or 'symm').")


def _iDivUp(a, b):
    return (a // b + 1) if (a % b != 0) else (a // b)


@cuda.jit(fastmath=True)
def _correlate2d_odd(inp, inpW, inpH, kernel, S, out, outW, outH):

    x, y = cuda.grid(2)
    j = x + S
    i = y + S

    if ((x >= 0) and (x < outW) and (y >= 0) and (y < outH)):
        oPixelPos = (y, x)
        temp: out.dtype = 0

        for k in range(-S, S + 1):
            for l in range(-S, S + 1):
                iPixelPos = ((i + k), (j + l))
                coefPos = ((k + S), (l + S))
                temp += inp[iPixelPos] * kernel[coefPos]

        out[oPixelPos] = temp


# For square/even kernels
@cuda.jit(fastmath=True)
def _correlate2d_even(inp, inpW, inpH, kernel, S, out, outW, outH):

    x, y = cuda.grid(2)
    j = x + S
    i = y + S

    if ((x >= 0) and (x < outW) and (y >= 0) and (y < outH)):
        oPixelPos = (y, x)
        temp: out.dtype = 0

        for k in range(-S, S):
            for l in range(-S, S):
                iPixelPos = ((i + k), (j + l))
                coefPos = (k + S, l + S)
                temp += inp[iPixelPos] * kernel[coefPos]

        out[oPixelPos] = temp


# For non-square kernels
@cuda.jit(fastmath=True)
def _correlate2d_ns(inp, inpW, inpH, kernel, S, S1, out, outW, outH):

    x, y = cuda.grid(2)
    j = x + S
    i = y + S1

    if ((x >= 0) and (x < outW) and (y >= 0) and (y < outH)):
        oPixelPos = (y, x)
        temp: out.dtype = 0

        for k in range(S):
            for l in range(S1):
                iPixelPos = ((i + k - S1), (j + l - S))
                coefPos = (k, l)
                temp += inp[iPixelPos] * kernel[coefPos]

        out[oPixelPos] = temp


# For square/odd kernels
@cuda.jit(fastmath=True)
def _convolve2d_odd(inp, inpW, inpH, kernel, S, out, outW, outH):

    x, y = cuda.grid(2)
    j = x + S
    i = y + S

    if ((x >= 0) and (x < outW) and (y >= 0) and (y < outH)):
        oPixelPos = (y, x)
        temp: out.dtype = 0

        for k in range(-S, S + 1):
            for l in range(-S, S + 1):
                iPixelPos = ((i + k), (j + l))
                coefPos = ((-k + S), (-l + S))
                temp += inp[iPixelPos] * kernel[coefPos]

        out[oPixelPos] = temp


# For square/even kernels
@cuda.jit(fastmath=True)
def _convolve2d_even(inp, inpW, inpH, kernel, S, out, outW, outH):

    x, y = cuda.grid(2)
    j = x + S
    i = y + S

    if ((x >= 0) and (x < outW) and (y >= 0) and (y < outH)):
        oPixelPos = (y, x)
        temp: out.dtype = 0

        for k in range(-S, S):
            for l in range(-S, S):
                iPixelPos = ((i + k), (j + l))
                coefPos = (-k + (S - 1), -l + (S - 1))
                temp += inp[iPixelPos] * kernel[coefPos]

        out[oPixelPos] = temp


# For non-square kernels
@cuda.jit(fastmath=True)
def _convolve2d_ns(inp, inpW, inpH, kernel, S, S1, out, outW, outH):

    x, y = cuda.grid(2)
    j = x + S
    i = y + S1

    if ((x >= 0) and (x < outW) and (y >= 0) and (y < outH)):
        oPixelPos = (y, x)
        temp: out.dtype = 0

        for k in range(S):
            for l in range(S1):
                iPixelPos = ((i + k - S1), (j + l - S))
                coefPos = (-k + (S - 1), -l + (S1 - 1))
                temp += inp[iPixelPos] * kernel[coefPos]

        out[oPixelPos] = temp


def _convolve2d_gpu(inp, out, kernel, mode, boundary, flip, fillvalue):

    if ((boundary != PAD) and (boundary != REFLECT) and
            (boundary != CIRCULAR)):
        raise Exception('Invalid boundary flag')

    # If kernel is square and odd
    if (kernel.shape[0] == kernel.shape[1]):  # square
        if (kernel.shape[0] % 2 == 1):  # odd
            pick = 1
            S = (kernel.shape[0] - 1) // 2
            if (mode == 2):  # full
                P1 = P2 = P3 = P4 = S * 2
            else:  # same/valid
                P1 = P2 = P3 = P4 = S
        else:  # even
            pick = 2
            S = kernel.shape[0] // 2
            if (mode == 2):  # full
                P1 = P2 = P3 = P4 = S * 2 - 1
            else:  # same/valid
                if (flip):
                    P1 = P2 = P3 = P4 = S
                else:
                    P1 = P3 = S - 1
                    P2 = P4 = S
    else:  # Non-square
        pick = 3
        S = kernel.shape[0]
        S1 = kernel.shape[1]
        if (mode == 2):  # full
            P1 = S - 1
            P2 = S - 1
            P3 = S1 - 1
            P4 = S1 - 1
        else:  # same/valid
            if (flip):
                P1 = S // 2
                P2 = S // 2 if (S % 2) else S // 2 - 1
                P3 = S1 // 2
                P4 = S1 // 2 if (S1 % 2) else S1 // 2 - 1
            else:
                P1 = S // 2 if (S % 2) else S // 2 - 1
                P2 = S // 2
                P3 = S1 // 2 if (S1 % 2) else S1 // 2 - 1
                P4 = S1 // 2

    if (mode == 1):  # SAME
        pad = ((P1, P2), (P3, P4))  # 4x5
        if (boundary == REFLECT):
            # symmetric not implemented in cupy, move to numpy
            inp = cp.asarray(np.pad(cp.asnumpy(inp), cp.asnumpy(pad),
                                    'symmetric'))
        if (boundary == CIRCULAR):
            inp = cp.asarray(np.pad(cp.asnumpy(inp), cp.asnumpy(pad), 'wrap'))
        if (boundary == PAD):
            inp = cp.pad(inp, pad, 'constant', constant_values=(fillvalue))

    if (mode == 2):  # FULL
        pad = ((P1, P2), (P3, P4))
        if (boundary == REFLECT):
            inp = cp.asarray(np.pad(cp.asnumpy(inp), cp.asnumpy(pad),
                                    'symmetric'))
        if (boundary == CIRCULAR):
            inp = cp.asarray(np.pad(cp.asnumpy(inp), cp.asnumpy(pad), 'wrap'))
        if (boundary == PAD):
            inp = cp.pad(inp, pad, 'constant', constant_values=(fillvalue))

    paddedW = inp.shape[1]
    paddedH = inp.shape[0]

    outW = out.shape[1]
    outH = out.shape[0]

    d_inp = cp.array(inp)
    d_kernel = cp.array(kernel)

    threadsPerBlock = (16, 16)
    blocksPerGrid = (_iDivUp(outW, threadsPerBlock[0]),
                     _iDivUp(outH, threadsPerBlock[1]))

    if (flip):
        if (pick == 1):
            _convolve2d_odd[blocksPerGrid, threadsPerBlock](
                d_inp, paddedW, paddedH, d_kernel, S, out, outW, outH
            )
        elif (pick == 2):
            _convolve2d_even[blocksPerGrid, threadsPerBlock](
                d_inp, paddedW, paddedH, d_kernel, S, out, outW, outH
            )
        elif (pick == 3):
            _convolve2d_ns[blocksPerGrid, threadsPerBlock](
                d_inp, paddedW, paddedH, d_kernel, S, S1, out, outW, outH
            )
    else:
        if (pick == 1):
            _correlate2d_odd[blocksPerGrid, threadsPerBlock](
                d_inp, paddedW, paddedH, d_kernel, S, out, outW, outH
            )
        elif (pick == 2):
            _correlate2d_even[blocksPerGrid, threadsPerBlock](
                d_inp, paddedW, paddedH, d_kernel, S, out, outW, outH
            )
        elif (pick == 3):
            _correlate2d_ns[blocksPerGrid, threadsPerBlock](
                d_inp, paddedW, paddedH, d_kernel, S, S1, out, outW, outH
            )

    return out


def _convolve2d(in1, in2, flip, mode='full', boundary='fill', fillvalue=0):

    # Promote inputs
    promType = cp.promote_types(in1.dtype, in2.dtype)
    in1 = in1.astype(promType)
    in2 = in2.astype(promType)

    if ((boundary != PAD) and (boundary != REFLECT) and
            (boundary != CIRCULAR)):
        raise Exception('Incorrect boundary value.')

    if ((boundary == PAD) and (fillvalue is not None)):
        fill = np.array(fillvalue, in1.dtype)
        if (fill is None):
            raise Exception('If you see this let developers know.')
        if (fill.size != 1):
            if (fill.size == 0):
                raise Exception('`fillvalue` cannot be an empty array.')
            raise Exception(
                '`fillvalue` must be scalar or an array with one element'
            )
    else:
        fill = np.zeros(1, in1.dtype)
        if (fill is None):
            raise Exception('Unable to create fill array')

    # Create empty array to hold number of aout dimensions
    out_dimens = np.empty(in1.ndim, np.int)
    if (mode == VALID):
        for i in range(in1.ndim):
            out_dimens[i] = in1.shape[i] - in2.shape[i] + 1
            if out_dimens[i] < 0:
                raise Exception(
                    'no part of the output is valid, use option 1 (same) or 2 \
                     (full) for third argument'
                )
    elif (mode == SAME):
        for i in range(in1.ndim):
            out_dimens[i] = in1.shape[i]
    elif (mode == FULL):
        for i in range(in1.ndim):
            out_dimens[i] = in1.shape[i] + in2.shape[i] - 1
    else:
        raise Exception('mode must be 0 (valid), 1 (same), or 2 (full)')

    # Create empty array out on GPU
    out = cp.empty(out_dimens.tolist(), in1.dtype)

    out = _convolve2d_gpu(
        in1,
        out,
        in2,
        mode,
        boundary,
        flip,
        fill)

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
    The functions `scipy.signal.get_window` and `scipy.signal.firwin`
    are called to generate the appropriate filter coefficients.

    The returned array of coefficients will always be of data type
    `complex128` to maintain precision. For use in lower-precision
    filter operations, this array should be converted to the desired
    data type before providing it to `cusignal.resample_poly`.

    """

    # Determine our up and down factors
    # Use a rational approimation to save computation time on really long
    # signals
    g_ = math.gcd(up, down)
    up //= g_
    down //= g_

    # Design a linear-phase low-pass FIR filter
    max_rate = max(up, down)
    f_c = 1.0 / max_rate  # cutoff of FIR filter (rel. to Nyquist)

    # reasonable cutoff for our sinc-like function
    half_len = 10 * max_rate

    h = firwin(2 * half_len + 1, f_c, window=window)
    return h
