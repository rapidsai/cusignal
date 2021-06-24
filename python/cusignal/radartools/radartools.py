# Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
from math import log2, ceil
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
    fft_template = cp.conj(cp.tile(cp.fft.fft(template, nfft),
                                   (num_pulses, 1)))
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

        pd_dataMatrix = \
            cp.fft.fft(cp.multiply(x,
                                   cp.tile(W.T, (1, samples_per_pulse)),
                                   nfft, axis=0))
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


def ambgfun(x, fs, prf, y=None, cut='2d', cutValue=0):
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

    if 'float64' in x.dtype.name:
        x = cp.asarray(x, dtype=cp.complex128)
    elif 'float32' in x.dtype.name:
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
    nfreq = 2**ceil(log2(len_seq - 1))

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

    if cut == '2d':
        new_ynorm = cp.empty((len_seq - 1, xlen), dtype=xnorm.dtype)
        _new_ynorm_kernel(xlen, xnorm, ynorm, new_ynorm)

        amf = nfreq * cp.abs(cp.fft.fftshift(
            cp.fft.ifft(new_ynorm, nfreq, axis=1), axes=1))

    elif cut == 'delay':
        Fd = cp.arange(-fs / 2, fs / 2, fs / nfreq)
        fftx = cp.fft.fft(xnorm, nfreq) * \
            cp.exp(1j * 2 * cp.pi * Fd * cutValue)
        xshift = cp.fft.ifft(fftx)

        ynorm_pad = cp.zeros(nfreq) + cp.zeros(nfreq) * 1j
        ynorm_pad[:ynorm.shape[0]] = ynorm

        amf = nfreq * cp.abs(cp.fft.ifftshift(
            cp.fft.ifft(ynorm_pad * cp.conj(xshift), nfreq)))

    elif cut == 'doppler':
        t = cp.arange(0, xlen) / fs
        ffty = cp.fft.fft(ynorm, len_seq - 1)
        fftx = cp.fft.fft(xnorm * cp.exp(1j * 2 * cp.pi * cutValue * t),
                          len_seq - 1)

        amf = cp.abs(cp.fft.fftshift(cp.fft.ifft(ffty * cp.conj(fftx))))

    else:
        raise ValueError('2d, delay, and doppler are the only\
            cut values allowed')

    return amf
