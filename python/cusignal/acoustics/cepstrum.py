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
import cupyx.scipy.fftpack as fft
import math


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
    ceps = fft.ifft(cp.log(cp.abs(spectrum)), n=n, axis=axis).real

    return ceps


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

    def _wrap(phase, ndelay):
        ndelay = cp.array(ndelay)
        samples = phase.shape[-1]
        center = (samples + 1) // 2
        wrapped = phase + cp.pi * ndelay[..., None] * cp.arange(samples) / center
        return wrapped

    log_spectrum = fft.fft(ceps)
    spectrum = cp.exp(log_spectrum.real + 1j * _wrap(log_spectrum.imag, ndelay))
    x = fft.ifft(spectrum).real
    return x