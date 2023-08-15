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

from math import ceil, log2

import cupy as cp

from ..windows.windows import get_window


def pulse_compression(x, template, normalize=False, window=None, nfft=None):
    """
    Pulse Compression is used to increase the range resolution and SNR
    by performing matched filtering of the transmitted pulse (template)
    with the received signal (x)

    Parameters
    ----------
    x : ndarray
        Received signal, assume 2D array with [num_pulses, sample_per_pulse]

    template : ndarray
        Transmitted signal, assume 1D array

    normalize : bool
        Normalize transmitted signal

    window : array_like, callable, string, float, or tuple, optional
        Specifies the window applied to the signal in the Fourier
        domain.

    nfft : int, size of FFT for pulse compression. Default is number of
        samples per pulse

    Returns
    -------
    compressedIQ : ndarray
        Pulse compressed output
    """
    [num_pulses, samples_per_pulse] = x.shape

    if nfft is None:
        nfft = samples_per_pulse

    if window is not None:
        Nx = len(template)
        if callable(window):
            W = window(cp.fft.fftfreq(Nx))
        elif isinstance(window, cp.ndarray):
            if window.shape != (Nx,):
                raise ValueError("window must have the same length as data")
            W = window
        else:
            W = get_window(window, Nx, False)

        template = cp.multiply(template, W)

    if normalize is True:
        template = cp.divide(template, cp.linalg.norm(template))

    fft_x = cp.fft.fft(x, nfft)
    fft_template = cp.conj(cp.tile(cp.fft.fft(template, nfft), (num_pulses, 1)))
    compressedIQ = cp.fft.ifft(cp.multiply(fft_x, fft_template), nfft)

    return compressedIQ


def pulse_doppler(x, window=None, nfft=None):
    """
    Pulse doppler processing yields a range/doppler data matrix that represents
    moving target data that's separated from clutter. An estimation of the
    doppler shift can also be obtained from pulse doppler processing. FFT taken
    across slow-time (pulse) dimension.

    Parameters
    ----------
    x : ndarray
        Received signal, assume 2D array with [num_pulses, sample_per_pulse]

    window : array_like, callable, string, float, or tuple, optional
        Specifies the window applied to the signal in the Fourier
        domain.

    nfft : int, size of FFT for pulse compression. Default is number of
        samples per pulse

    Returns
    -------
    pd_dataMatrix : ndarray
        Pulse-doppler output (range/doppler matrix)
    """
    [num_pulses, samples_per_pulse] = x.shape

    if nfft is None:
        nfft = num_pulses

    if window is not None:
        Nx = num_pulses
        if callable(window):
            W = window(cp.fft.fftfreq(Nx))
        elif isinstance(window, cp.ndarray):
            if window.shape != (Nx,):
                raise ValueError("window must have the same length as data")
            W = window
        else:
            W = get_window(window, Nx, False)[cp.newaxis]

        pd_dataMatrix = cp.fft.fft(
            cp.multiply(x, cp.tile(W.T, (1, samples_per_pulse)), nfft, axis=0)
        )
    else:
        pd_dataMatrix = cp.fft.fft(x, nfft, axis=0)

    return pd_dataMatrix


_new_ynorm_kernel = cp.ElementwiseKernel(
    "int32 xlen, raw T xnorm, raw T ynorm",
    "T out",
    """
    int row = i / xlen;
    int col = i % xlen;
    int x_col = col - ( xlen - 1 ) + row;

    if ( ( x_col >= 0 ) && ( x_col < xlen ) ) {
        out = ynorm[col] * thrust::conj( xnorm[x_col] );
    } else {
        out = T(0,0);
    }
    """,
    "_new_ynorm_kernel",
    options=("-std=c++11",),
)


def ambgfun(x, fs, prf, y=None, cut="2d", cutValue=0):
    """
    Calculates the normalized ambiguity function for the vector x

    Parameters
    ----------
    x : ndarray
        Input pulse waveform

    fs: int, float
        Sampling rate in Hz

    prf: int, float
        Pulse repetition frequency in Hz

    y : ndarray
        Second input pulse waveform. If not given, y = x

    cut : string
        Direction of one-dimensional cut through ambiguity function

    cutValue : int, float
        Time delay or doppler shift at which one-dimensional cut
        through ambiguity function is taken

    Returns
    -------
    amfun : ndarray
        Normalized magnitude of the ambiguity function
    """
    cut = cut.lower()

    if "float64" in x.dtype.name:
        x = cp.asarray(x, dtype=cp.complex128)
    elif "float32" in x.dtype.name:
        x = cp.asarray(x, dtype=cp.complex64)
    else:
        x = cp.asarray(x)

    xnorm = x / cp.linalg.norm(x)
    if y is None:
        y = x
        ynorm = xnorm
    else:
        ynorm = y / cp.linalg.norm(y)

    len_seq = len(xnorm) + len(ynorm)
    nfreq = 2 ** ceil(log2(len_seq - 1))

    # Consider for deletion as we add different cut values
    """
    if len(xnorm) < len(ynorm):
        len_diff = len(ynorm) - len(xnorm)
        ynorm = cp.concatenate(ynorm, cp.zeros(len_diff))
    elif len(xnorm) > len(ynorm):
        len_diff = len(xnorm) - len(ynorm)
        xnorm = cp.concatenate(xnorm, cp.zeros(len_diff))
    """

    xlen = len(xnorm)

    if cut == "2d":
        new_ynorm = cp.empty((len_seq - 1, xlen), dtype=xnorm.dtype)
        _new_ynorm_kernel(xlen, xnorm, ynorm, new_ynorm)

        amf = nfreq * cp.abs(
            cp.fft.fftshift(cp.fft.ifft(new_ynorm, nfreq, axis=1), axes=1)
        )

    elif cut == "delay":
        Fd = cp.arange(-fs / 2, fs / 2, fs / nfreq)
        fftx = cp.fft.fft(xnorm, nfreq) * cp.exp(1j * 2 * cp.pi * Fd * cutValue)
        xshift = cp.fft.ifft(fftx)

        ynorm_pad = cp.zeros(nfreq) + cp.zeros(nfreq) * 1j
        ynorm_pad[: ynorm.shape[0]] = ynorm

        amf = nfreq * cp.abs(
            cp.fft.ifftshift(cp.fft.ifft(ynorm_pad * cp.conj(xshift), nfreq))
        )

    elif cut == "doppler":
        t = cp.arange(0, xlen) / fs
        ffty = cp.fft.fft(ynorm, len_seq - 1)
        fftx = cp.fft.fft(xnorm * cp.exp(1j * 2 * cp.pi * cutValue * t), len_seq - 1)

        amf = cp.abs(cp.fft.fftshift(cp.fft.ifft(ffty * cp.conj(fftx))))

    else:
        raise ValueError(
            "2d, delay, and doppler are the only\
            cut values allowed"
        )

    return amf


def cfar_alpha(pfa, N):
    """
    Computes the value of alpha corresponding to a given probability
    of false alarm and number of reference cells N.

    Parameters
    ----------
    pfa : float
        Probability of false alarm.

    N : int
        Number of reference cells.

    Returns
    -------
    alpha : float
        Alpha value.
    """
    return N * (pfa ** (-1.0 / N) - 1)


def ca_cfar(array, guard_cells, reference_cells, pfa=1e-3):
    """
    Computes the cell-averaged constant false alarm rate (CA CFAR) detector
    threshold and returns for a given array.

    Parameters
    ----------
    array : ndarray
        Array containing data to be processed.

    guard_cells_x : int
        One-sided guard cell count in the first dimension.

    guard_cells_y : int
        One-sided guard cell count in the second dimension.

    reference_cells_x : int
        one-sided reference cell count in the first dimension.

    reference_cells_y : int
        one-sided referenc cell count in the second dimension.

    pfa : float
        Probability of false alarm.

    Returns
    -------
    threshold : ndarray
        CFAR threshold
    return : ndarray
        CFAR detections
    """
    shape = array.shape
    if len(shape) > 2:
        raise TypeError("Only 1D and 2D arrays are currently supported.")
    mask = cp.zeros(shape, dtype=cp.float32)

    if len(shape) == 1:
        if len(array) <= 2 * guard_cells + 2 * reference_cells:
            raise ValueError("Array too small for given parameters")
        intermediate = cp.cumsum(array, axis=0, dtype=cp.float32)
        N = 2 * reference_cells
        alpha = cfar_alpha(pfa, N)
        tpb = (32,)
        bpg = (
            (len(array) - 2 * reference_cells - 2 * guard_cells + tpb[0] - 1) // tpb[0],
        )
        _ca_cfar_1d_kernel(
            bpg,
            tpb,
            (
                array,
                intermediate,
                mask,
                len(array),
                N,
                cp.float32(alpha),
                guard_cells,
                reference_cells,
            ),
        )
    elif len(shape) == 2:
        if len(guard_cells) != 2 or len(reference_cells) != 2:
            raise TypeError("Guard and reference cells must be two " "dimensional.")
        guard_cells_x, guard_cells_y = guard_cells
        reference_cells_x, reference_cells_y = reference_cells
        if shape[0] - 2 * guard_cells_x - 2 * reference_cells_x <= 0:
            raise ValueError("Array first dimension too small for given " "parameters.")
        if shape[1] - 2 * guard_cells_y - 2 * reference_cells_y <= 0:
            raise ValueError(
                "Array second dimension too small for given " "parameters."
            )
        intermediate = cp.cumsum(array, axis=0, dtype=cp.float32)
        intermediate = cp.cumsum(intermediate, axis=1, dtype=cp.float32)
        N = 2 * reference_cells_x * (2 * reference_cells_y + 2 * guard_cells_y + 1)
        N += 2 * (2 * guard_cells_x + 1) * reference_cells_y
        alpha = cfar_alpha(pfa, N)
        tpb = (8, 8)
        bpg_x = (
            shape[0] - 2 * (reference_cells_x + guard_cells_x) + tpb[0] - 1
        ) // tpb[0]
        bpg_y = (
            shape[1] - 2 * (reference_cells_y + guard_cells_y) + tpb[1] - 1
        ) // tpb[1]
        bpg = (bpg_x, bpg_y)
        _ca_cfar_2d_kernel(
            bpg,
            tpb,
            (
                array,
                intermediate,
                mask,
                shape[0],
                shape[1],
                N,
                cp.float32(alpha),
                guard_cells_x,
                guard_cells_y,
                reference_cells_x,
                reference_cells_y,
            ),
        )
    return (mask, array - mask > 0)


_ca_cfar_2d_kernel = cp.RawKernel(
    r"""
extern "C" __global__ void
_ca_cfar_2d_kernel(float * array, float * intermediate, float * mask,
                   int width, int height, int N, float alpha,
                   int guard_cells_x, int guard_cells_y,
                   int reference_cells_x, int reference_cells_y)
{
    int i_init = threadIdx.x+blockIdx.x*blockDim.x;
    int j_init = threadIdx.y+blockIdx.y*blockDim.y;
    int i, j, x, y, offset;
    int tro, tlo, blo, bro, tri, tli, bli, bri;
    float outer_area, inner_area, T;

    for (i=i_init; i<width-2*(guard_cells_x+reference_cells_x);
         i += blockDim.x*gridDim.x){
        for (j=j_init; j<height-2*(guard_cells_y+reference_cells_y);
             j += blockDim.y*gridDim.y){
            /* 'tri' is Top Right Inner (square), 'blo' is Bottom Left
             * Outer (square), etc. These are the corners at which
             * the intermediate array must be evaluated.
             */
            x = i+guard_cells_x+reference_cells_x;
            y = j+guard_cells_y+reference_cells_y;
            offset = x*height+y;

            tro = (x+guard_cells_x+reference_cells_x)*height+y+
                guard_cells_y+reference_cells_y;
            tlo = (x-guard_cells_x-reference_cells_x-1)*height+y+
                guard_cells_y+reference_cells_y;
            blo = (x-guard_cells_x-reference_cells_x-1)*height+y-
                guard_cells_y-reference_cells_y-1;
            bro = (x+guard_cells_x+reference_cells_x)*height+y-
                guard_cells_y-reference_cells_y-1;

            tri = (x+guard_cells_x)*height+y+guard_cells_y;
            tli = (x-guard_cells_x-1)*height+y+guard_cells_y;
            bli = (x-guard_cells_x-1)*height+y-guard_cells_y-1;
            bri = (x+guard_cells_x)*height+y-guard_cells_y-1;

            /* It would be nice to eliminate the triple
             * branching here, but it only occurs on the boundaries
             * of the array (i==0 or j==0). So it shouldn't hurt
             * overall performance much.
             */
            if (i>0 && j>0){
                outer_area = intermediate[tro]-intermediate[tlo]-
                    intermediate[bro]+intermediate[blo];
            } else if (i == 0 && j > 0){
                outer_area = intermediate[tro]-intermediate[bro];
            } else if (i > 0 && j == 0){
                outer_area = intermediate[tro]-intermediate[tlo];
            } else if (i == 0 && j == 0){
                outer_area = intermediate[tro];
            }

            inner_area = intermediate[tri]-intermediate[tli]-
                intermediate[bri]+intermediate[bli];

            T = outer_area-inner_area;
            T = alpha/N*T;
            mask[offset] = T;
        }
    }
}
""",
    "_ca_cfar_2d_kernel",
)


_ca_cfar_1d_kernel = cp.RawKernel(
    r"""
extern "C" __global__ void
_ca_cfar_1d_kernel(float * array, float * intermediate, float * mask,
                   int width, int N, float alpha,
                   int guard_cells, int reference_cells)
{
    int i_init = threadIdx.x+blockIdx.x*blockDim.x;
    int i, x;
    int br, bl, sr, sl;
    float big_area, small_area, T;

    for (i=i_init; i<width-2*(guard_cells+reference_cells);
         i += blockDim.x*gridDim.x){
        x = i+guard_cells+reference_cells;

        br = x+guard_cells+reference_cells;
        bl = x-guard_cells-reference_cells-1;
        sr = x+guard_cells;
        sl = x-guard_cells-1;

        if (i>0){
            big_area = intermediate[br]-intermediate[bl];
        } else{
            big_area = intermediate[br];
        }
        small_area = intermediate[sr]-intermediate[sl];

        T = big_area-small_area;
        T = alpha/N*T;
        mask[x] = T;
    }
}
""",
    "_ca_cfar_1d_kernel",
)
