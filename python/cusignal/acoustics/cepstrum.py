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
import cupyx.scipy.fftpack as fft
import math


_real_cepstrum_kernel = cp.ElementwiseKernel(
    "T spectrum",
    "T output",
    """
    output = log( abs( spectrum ) );
    """,
    "_real_cepstrum_kernel",
)


def real_cepstrum(x, n=None, axis=-1):
    r"""
    Calculates the real cepstrum of an input sequence x where the cepstrum is
    defined as the inverse Fourier transform of the log magnitude DFT
    (spectrum) of a signal. It's primarily used for source/speaker separation
    in speech signal processing

    Parameters
    ----------
    x : ndarray
        Input sequence, if x is a matrix, return cepstrum in direction of axis
    n : int
        Size of Fourier Transform; If none, will use length of input array
    axis: int
        Direction for cepstrum calculation

    Returns
    -------
    ceps : ndarray
        Complex cepstrum result
    """
    x = cp.asarray(x)
    spectrum = fft.fft(x, n=n, axis=axis)
    spectrum = _real_cepstrum_kernel(spectrum)
    return fft.ifft(spectrum, n=n, axis=axis).real


_complex_cepstrum_kernel = cp.ElementwiseKernel(
    "float64 samples, T ndelay, float64 pi, int64 ar, T center, T unwrapped, complex128 spectrum",
    "complex128 log_spectrum",
    """
    T unwrapped_phase = unwrapped - pi * ndelay * ar / center;
    log_spectrum =  log( abs( spectrum ) ) * unwrapped_phase;
    """,
    "_complex_cepstrum_kernel",
)


def complex_cepstrum(x, n=None, axis=-1):
    r"""
    Calculates the complex cepstrum of a real valued input sequence x
    where the cepstrum is defined as the inverse Fourier transform
    of the log magnitude DFT (spectrum) of a signal. It's primarily
    used for source/speaker separation in speech signal processing.

    The input is altered to have zero-phase at pi radians (180 degrees)
    Parameters
    ----------
    x : ndarray
        Input sequence, if x is a matrix, return cepstrum in direction of axis
    n : int
       Size of Fourier Transform; If none, will use length of input array
    axis: int
        Direction for cepstrum calculation
    Returns
    -------
    ceps : ndarray
        Complex cepstrum result
    """
    x = cp.asarray(x)

    def _unwrap(x):
        r"""
        Unwrap phase for complex cepstrum calculation; helper function
        """

        samples = len(x)
        unwrapped = cp.unwrap(x)
        center = math.floor((samples + 1) / 2)
        ndelay = cp.round_(unwrapped[center] / cp.pi)
        unwrapped -= cp.pi * ndelay * cp.arange(samples) / center

        return unwrapped, ndelay

    spectrum = fft.fft(x, n=n, axis=axis)
    unwrapped_phase, ndelay = _unwrap(cp.angle(spectrum))
    log_spectrum = cp.log(cp.abs(spectrum)) + 1j * unwrapped_phase
    ceps = fft.ifft(log_spectrum, n=n, axis=axis).real

    return ceps, ndelay

    # spectrum = fft.fft(x, n=n, axis=axis)
    # ang_spec = cp.angle(spectrum)
    # unwrapped = cp.unwrap(ang_spec)
    # samples = len(ang_spec)
    # center = math.floor((samples + 1) / 2)
    # ndelay = cp.round_(unwrapped[center] / cp.pi)
    # ar = cp.arange(samples)

    # log_spectrum = _complex_cepstrum_kernel(samples, ndelay, np.pi, ar, center, unwrapped, spectrum)
    # ceps = fft.ifft(log_spectrum, n=n, axis=axis).real
    # return ceps, ndelay


_inverse_complex_cepstrum_kernel = cp.ElementwiseKernel(
    "complex128 log_spectrum, int64 ndelay, float64 pi",
    "complex128 spectrum",
    """
    int center = ( ( sizeof( imag( log_spectrum ) ) + 1 ) / 2 );
    spectrum = imag( log_spectrum ) + pi * ndelay * sizeof( imag( log_spectrum ) ) / center;
    spectrum = exp( real(log_spectrum) + imag( spectrum ))
    """,
    "_inverse_complex_cepstrum_kernel",
)


def inverse_complex_cepstrum(ceps, ndelay):
    r"""Compute the inverse complex cepstrum of a real sequence.
    ceps : ndarray
        Real sequence to compute inverse complex cepstrum of.
    ndelay: int
        The amount of samples of circular delay added to `x`.
    Returns
    -------
    x : ndarray
        The inverse complex cepstrum of the real sequence `ceps`.
    The inverse complex cepstrum is given by
    .. math:: x[n] = F^{-1}\left{\exp(F(c[n]))\right}
    where :math:`c_[n]` is the input signal and :math:`F` and :math:`F_{-1}
    are respectively the forward and backward Fourier transform.
    """
    log_spectrum = fft.fft(ceps)
    spectrum = _inverse_complex_cepstrum_kernel(log_spectrum, ndelay, np.pi)
    x = fft.ifft(spectrum).real
    return x


def minimum_phase(x, n=None):
    r"""Compute the minimum phase reconstruction of a real sequence.
    x : ndarray
        Real sequence to compute the minimum phase reconstruction of.
    n : {None, int}, optional
        Length of the Fourier transform.
    Compute the minimum phase reconstruction of a real sequence using the
    real cepstrum.
    Returns
    -------
    m : ndarray
        The minimum phase reconstruction of the real sequence `x`.
    """
    if n is None:
        n = len(x)
    ceps = real_cepstrum(x, n=n)
    odd = n % 2

    window = cp.concatenate(
        (
            cp.array([1.0]),
            2.0 * cp.ones((n + odd) // 2 - 1),
            cp.ones(1 - odd),
            cp.zeros((n + odd) // 2 - 1),
        )
    )
    m = fft.ifft(cp.exp(fft.fft(window * ceps))).real

    return m
