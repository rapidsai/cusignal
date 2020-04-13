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
from string import Template

import numpy as np
import cupy as cp
from numba import (
    complex64,
    complex128,
    cuda,
    float32,
    float64,
    int32,
    int64,
    void,
)
try:
    # Numba <= 0.49
    from numba.types.scalars import Complex
except ImportError:
    # Numba >= 0.49
    from numba.core.types.scalars import Complex

from .fir_filter_design import firwin

_numba_kernel_cache = {}
_cupy_kernel_cache = {}

# Numba type supported and corresponding C type
_SUPPORTED_TYPES = {
    int32: "int",
    int64: "long int",
    float32: "float",
    float64: "double",
    complex64: "complex<float>",
    complex128: "complex<double>",
}

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


# Use until functionality provided in Numba 0.49/0.50 available
def stream_cupy_to_numba(cp_stream):
    """
    Notes:
        1. The lifetime of the returned Numba stream should be as
           long as the CuPy one, which handles the deallocation
           of the underlying CUDA stream.
        2. The returned Numba stream is assumed to live in the same
           CUDA context as the CuPy one.
        3. The implementation here closely follows that of
           cuda.stream() in Numba.
    """
    from ctypes import c_void_p
    import weakref

    # get the pointer to actual CUDA stream
    raw_str = cp_stream.ptr

    # gather necessary ingredients
    ctx = cuda.devices.get_context()
    handle = c_void_p(raw_str)

    # create a Numba stream
    nb_stream = cuda.cudadrv.driver.Stream(
        weakref.proxy(ctx), handle, finalizer=None
    )

    return nb_stream


def _numba_correlate_2d(
    inp, inpW, inpH, kernel, S0, S1, out, outW, outH, pick
):

    y, x = cuda.grid(2)

    if pick != 3:  # non-square
        i = cp.int32(x + S0)
    else:
        i = cp.int32(x + S1)
    j = cp.int32(y + S0)

    oPixelPos = (x, y)
    if (x < outH) and (y < outW):

        temp: out.dtype = 0

        if pick == 1:  # odd
            for k in range(cp.int32(-S0), cp.int32(S0 + 1)):
                for l in range(cp.int32(-S0), cp.int32(S0 + 1)):
                    iPixelPos = (cp.int32(i + k), cp.int32(j + l))
                    coefPos = (cp.int32(k + S0), cp.int32(l + S0))
                    temp += inp[iPixelPos] * kernel[coefPos]
                    # temp = x

        elif pick == 2:  # even
            for k in range(cp.int32(-S0), cp.int32(S0)):
                for l in range(cp.int32(-S0), cp.int32(S0)):
                    iPixelPos = (cp.int32(i + k), cp.int32(j + l))
                    coefPos = (cp.int32(k + S0), cp.int32(l + S0))
                    temp += inp[iPixelPos] * kernel[coefPos]

        else:  # non-squares
            for k in range(cp.int32(S0)):
                for l in range(cp.int32(S1)):
                    iPixelPos = (
                        cp.int32(cp.int32(i + k) - S1),
                        cp.int32(cp.int32(j + l) - S0),
                    )
                    coefPos = (k, l)
                    temp += inp[iPixelPos] * kernel[coefPos]

        out[oPixelPos] = temp


def _numba_convolve_2d(inp, inpW, inpH, kernel, S0, S1, out, outW, outH, pick):

    y, x = cuda.grid(2)

    if pick != 3:  # non-square
        i = cp.int32(x + S0)
    else:
        i = cp.int32(x + S1)
    j = cp.int32(y + S0)

    oPixelPos = (x, y)
    if (x < outH) and (y < outW):

        temp: out.dtype = 0

        if pick == 1:  # odd
            for k in range(cp.int32(-S0), cp.int32(S0 + 1)):
                for l in range(cp.int32(-S0), cp.int32(S0 + 1)):
                    iPixelPos = (cp.int32(i + k), cp.int32(j + l))
                    coefPos = (cp.int32(-k + S0), cp.int32(-l + S0))
                    temp += inp[iPixelPos] * kernel[coefPos]

        elif pick == 2:  # even
            for k in range(cp.int32(-S0), cp.int32(S0)):
                for l in range(cp.int32(-S0), cp.int32(S0)):
                    iPixelPos = (cp.int32(i + k), cp.int32(j + l))
                    coefPos = (
                        cp.int32(cp.int32(-k + S0) - 1),
                        cp.int32(cp.int32(-l + S0) - 1),
                    )
                    temp += inp[iPixelPos] * kernel[coefPos]

        else:  # non-squares
            for k in range(cp.int32(S0)):
                for l in range(cp.int32(S1)):
                    iPixelPos = (
                        cp.int32(cp.int32(i + k) - S1),
                        cp.int32(cp.int32(j + l) - S0),
                    )
                    coefPos = (
                        cp.int32(cp.int32(-k + S0) - 1),
                        cp.int32(cp.int32(-l + S1) - 1),
                    )
                    temp += inp[iPixelPos] * kernel[coefPos]

        out[oPixelPos] = temp


def _numba_convolve_2d_signature(ty):
    return void(
        ty[:, :],  # inp
        int64,  # inpW
        int64,  # inpH
        ty[:, :],  # kernel
        int64,  # S0
        int64,  # S1 - only used by non-squares
        ty[:, :],  # out
        int64,  # outW
        int64,  # outH
        int64,  # pick
    )


# Custom Cupy raw kernel implementing upsample, filter, downsample operation
# Matthew Nicely - mnicely@nvidia.com
loaded_from_source = Template(
    """
$header

extern "C" {
    __global__ void _cupy_correlate_2d(
            const ${datatype} * __restrict__ inp,
            const int inpW,
            const int inpH,
            const ${datatype} * __restrict__ kernel,
            const int kerW,
            const int kerH,
            const int S0,
            const int S1,
            ${datatype} * __restrict__ out,
            const int outW,
            const int outH,
            const int pick) {

        const int ty {
            static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) };
        const int tx {
            static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) };

        int i {};
        if ( pick != 3 ) {
            i = tx + S0;
        } else {
            i = tx + S1;
        }
        int j { ty + S0 };

        int2 oPixelPos {tx, ty};
        if ( (tx < outH) && (ty < outW) ) {
            ${datatype} temp {};

            // Odd
            if ( pick == 1) {
                for (int k = -S0; k < (S0 + 1); k++){
                    for (int l = -S0; l < (S0 + 1); l++) {
                        int2 iPixelPos {(i + k), (j + l)};
                        int2 coefPos {(k + S0), (l + S0)};
                        temp += inp[iPixelPos.x * inpW + iPixelPos.y] *
                            kernel[coefPos.x * kerW + coefPos.y];
                    }
                }

            // Even
            } else if (pick == 2) {
                for (int k = -S0; k < S0; k++){
                    for (int l = -S0; l < S0; l++) {
                        int2 iPixelPos {(i + k), (j + l)}; // iPixelPos[1], [0]
                        int2 coefPos {(k + S0), (l + S0)};
                        temp += inp[iPixelPos.x * inpW + iPixelPos.y] *
                            kernel[coefPos.x * kerW + coefPos.y];
                    }
                }

            // Non-squares
            } else {
                for (int k = 0; k < S0; k++){
                    for (int l = 0; l < S1; l++) {
                        int2 iPixelPos {(i + k - S1), (j + l - S0)};
                        int2 coefPos {k, l};
                        temp += inp[iPixelPos.x * inpW + iPixelPos.y] *
                            kernel[coefPos.x * kerH + coefPos.y];
                    }
                }
            }
            out[oPixelPos.x * outW + oPixelPos.y] = temp;
        }
    }

    __global__ void _cupy_convolve_2d(
            const ${datatype} * __restrict__ inp,
            const int inpW,
            const int inpH,
            const ${datatype} * __restrict__ kernel,
            const int kerW,
            const int kerH,
            const int S0,
            const int S1,
            ${datatype} * __restrict__ out,
            const int outW,
            const int outH,
            const int pick) {

        const int ty {
            static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) };
        const int tx {
            static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) };

        int i {};
        if ( pick != 3 ) {
            i = tx + S0;
        } else {
            i = tx + S1;
        }
        int j { ty + S0 };

        int2 oPixelPos {tx, ty};
        if ( (tx < outH) && (ty < outW) ) {
            ${datatype} temp {};

            // Odd kernel
            if ( pick == 1) {
                for (int k = -S0; k < (S0 + 1); k++){
                    for (int l = -S0; l < (S0 + 1); l++) {
                        int2 iPixelPos {(i + k), (j + l)};
                        int2 coefPos {(-k + S0), (-l + S0)};
                        temp += inp[iPixelPos.x * inpW + iPixelPos.y] *
                            kernel[coefPos.x * kerW + coefPos.y];
                    }
                }
            // Even kernel
            } else if (pick == 2) {
                for (int k = -S0; k < S0; k++){
                    for (int l = -S0; l < S0; l++) {
                        int2 iPixelPos {(i + k), (j + l)};
                        int2 coefPos {(-k + S0 - 1), (-l + S0 - 1)};
                        temp += inp[iPixelPos.x * inpW + iPixelPos.y] *
                            kernel[coefPos.x * kerW + coefPos.y];
                    }
                }

            // Non-squares kernel
            } else {
                for (int k = 0; k < S0; k++){
                    for (int l = 0; l < S1; l++) {
                        int2 iPixelPos {(i + k - S1), (j + l - S0)};
                        int2 coefPos {(-k + S0 - 1), (-l + S1 - 1)};
                        temp += inp[iPixelPos.x * inpW + iPixelPos.y] *
                            kernel[coefPos.x * kerH + coefPos.y];
                    }
                }
            }
            out[oPixelPos.x * outW + oPixelPos.y] = temp;
        }
    }
}
"""
)


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


def _get_backend_kernel(flip, dtype, grid, block, stream, use_numba):

    if use_numba:
        nb_stream = stream_cupy_to_numba(stream)
        kernel = _numba_kernel_cache[(flip, dtype.name)]

        if kernel:
            return kernel[grid, block, nb_stream]
    else:
        kernel = _cupy_kernel_cache[(flip, dtype.name)]
        if kernel:
            return _cupy_convolve_2d_wrapper(grid, block, stream, kernel)

    raise NotImplementedError(
        "No kernel found for flip {}, datatype {}".format(flip, dtype.name)
    )


def _populate_kernel_cache():
    for numba_type, c_type in _SUPPORTED_TYPES.items():
        # JIT compile the numba kernels, flip = 0/1 (correlate/convolve)
        sig = _numba_convolve_2d_signature(numba_type)
        _numba_kernel_cache[(0, str(numba_type))] = cuda.jit(
            sig, fastmath=True
        )(_numba_correlate_2d)
        _numba_kernel_cache[(1, str(numba_type))] = cuda.jit(
            sig, fastmath=True
        )(_numba_convolve_2d)

        # Instantiate the cupy kernel for this type and compile
        if isinstance(numba_type, Complex):
            header = "#include <cupy/complex.cuh>"
        else:
            header = ""
        src = loaded_from_source.substitute(datatype=c_type, header=header)
        module2 = cp.RawModule(
            code=src, options=("-std=c++11", "-use_fast_math")
        )
        _cupy_kernel_cache[(0, str(numba_type))] = module2.get_function(
            "_cupy_correlate_2d"
        )
        _cupy_kernel_cache[(1, str(numba_type))] = module2.get_function(
            "_cupy_convolve_2d"
        )


def _convolve2d_gpu(
    inp, out, kernel, mode, boundary, flip, fillvalue, cp_stream, use_numba,
):

    if (boundary != PAD) and (boundary != REFLECT) and (boundary != CIRCULAR):
        raise Exception("Invalid boundary flag")

    S = np.zeros(2, dtype=int)

    # If kernel is square and odd
    if kernel.shape[0] == kernel.shape[1]:  # square
        if kernel.shape[0] % 2 == 1:  # odd
            pick = 1
            S[0] = (kernel.shape[0] - 1) // 2
            if mode == 2:  # full
                P1 = P2 = P3 = P4 = S[0] * 2
            else:  # same/valid
                P1 = P2 = P3 = P4 = S[0]
        else:  # even
            pick = 2
            S[0] = kernel.shape[0] // 2
            if mode == 2:  # full
                P1 = P2 = P3 = P4 = S[0] * 2 - 1
            else:  # same/valid
                if flip:
                    P1 = P2 = P3 = P4 = S[0]
                else:
                    P1 = P3 = S[0] - 1
                    P2 = P4 = S[0]
    else:  # Non-square
        pick = 3
        S[0] = kernel.shape[0]
        S[1] = kernel.shape[1]
        if mode == 2:  # full
            P1 = S[0] - 1
            P2 = S[0] - 1
            P3 = S[1] - 1
            P4 = S[1] - 1
        else:  # same/valid
            if flip:
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
    d_kernel = cp.array(kernel)

    threadsperblock = (16, 16)
    blockspergrid = (
        _iDivUp(outW, threadsperblock[0]),
        _iDivUp(outH, threadsperblock[1]),
    )

    kernell = _get_backend_kernel(
        flip, out.dtype, blockspergrid, threadsperblock, cp_stream, use_numba,
    )

    kernell(
        d_inp, paddedW, paddedH, d_kernel, S[0], S[1], out, outW, outH, pick
    )
    return out


def _convolve2d(
    in1,
    in2,
    flip,
    mode="full",
    boundary="fill",
    fillvalue=0,
    cp_stream=cp.cuda.stream.Stream(null=True),
    use_numba=False,
):

    # Promote inputs
    promType = cp.promote_types(in1.dtype, in2.dtype)
    in1 = in1.astype(promType)
    in2 = in2.astype(promType)

    if (boundary != PAD) and (boundary != REFLECT) and (boundary != CIRCULAR):
        raise Exception("Incorrect boundary value.")

    if (boundary == PAD) and (fillvalue is not None):
        fill = np.array(fillvalue, in1.dtype)
        if fill is None:
            raise Exception("If you see this let developers know.")
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
    if mode == VALID:
        for i in range(in1.ndim):
            out_dimens[i] = in1.shape[i] - in2.shape[i] + 1
            if out_dimens[i] < 0:
                raise Exception(
                    "no part of the output is valid, use option 1 (same) or 2 \
                     (full) for third argument"
                )
    elif mode == SAME:
        for i in range(in1.ndim):
            out_dimens[i] = in1.shape[i]
    elif mode == FULL:
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
        mode,
        boundary,
        flip,
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


# 1) Load and compile upfirdn kernels for each supported data type.
_populate_kernel_cache()
