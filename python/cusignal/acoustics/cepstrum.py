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

from math import floor, pi

import cupy as cp
import cupyx.scipy.fftpack as fft


def rceps(x, n=None, axis=-1):
    """
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

    h = fft.fft(x, n=n, axis=axis)
    ceps = fft.ifft(cp.log(cp.abs(h)), n=n, axis=axis).real

    return ceps


def cceps_unwrap(x):
    """
    Unwrap phase for complex cepstrum calculation; helper function
    """
    x = cp.asarray(x)

    n = len(x)
    y = cp.unwrap(x)
    nh = floor((n + 1) / 2)
    nd = cp.round_(y[nh] / pi)
    y = y - cp.pi * nd * cp.arange(0, n) / nh

    return y


def cceps(x, n=None, axis=-1):
    """
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

    h = fft.fft(x, n=n, axis=axis)
    ah = cceps_unwrap(cp.angle(h))
    logh = cp.log(cp.abs(h)) + 1j * ah
    cceps = fft.ifft(logh, n=n, axis=axis).real

    return cceps


def _wrap(phase, ndelay):
    """
    Wrap phase for inverse complex cepstrum helper function
    """

    ndelay = cp.array(ndelay)
    samples = phase.shape[-1]
    center = (samples + 1) // 2
    wrapped = phase + cp.pi * ndelay[..., None] * cp.arange(samples) / center
    return wrapped


def inverse_complex_cepstrum(ceps, ndelay):
    """Compute the inverse complex cepstrum of a real sequence.
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
    See Also
    --------
    complex_cepstrum: Compute the complex cepstrum of a real sequence.
    real_cepstrum: Compute the real cepstrum of a real sequence.
    Examples
    --------
    Taking the complex cepstrum and then the inverse complex cepstrum results
    in the original sequence.
    >>> import numpy as np
    >>> from scipy.signal import inverse_complex_cepstrum
    >>> x = np.arange(10)
    >>> ceps, ndelay = complex_cepstrum(x)
    >>> y = inverse_complex_cepstrum(ceps, ndelay)
    >>> print(x)
    >>> print(y)
    References
    ----------
    .. [1] Wikipedia, "Cepstrum".
           http://en.wikipedia.org/wiki/Cepstrum
    """

    ceps = cp.asarray(ceps)

    log_spectrum = fft.fft(ceps)
    spectrum = cp.exp(
        log_spectrum.real + 1j * _wrap(log_spectrum.imag, ndelay)
    )
    x = fft.ifft(spectrum).real
    return x


def minimum_phase(x, n=None):
    """Compute the minimum phase reconstruction of a real sequence.
    x : ndarray
        Real sequence to compute the minimum phase reconstruction of.
    n : {None, int}, optional
        Length of the Fourier transform.
    Compute the minimum phase reconstruction of a real sequence using the
    real cepstrum.

    Compute the minimum phase reconstruction of a real sequence using the
    real cepstrum.
    Returns
    -------
    m : ndarray
        The minimum phase reconstruction of the real sequence `x`.
    See Also
    --------
    real_cepstrum: Compute the real cepstrum.
    Examples
    --------
    >>> from scipy.signal import minimum_phase
    References
    ----------
    .. [1] Soo-Chang Pei, Huei-Shan Lin. Minimum-Phase FIR Filter Design Using
           Real Cepstrum. IEEE TRANSACTIONS ON CIRCUITS AND SYSTEMS-II:
           EXPRESS BRIEFS, VOL. 53, NO. 10, OCTOBER 2006

    """

    x = cp.asarray(x)

    if n is None:
        n = len(x)
    ceps = rceps(x, n=n)
    odd = n % 2
    window = cp.concatenate(
        (
            [1.0],
            2.0 * cp.ones((n + odd) / 2 - 1),
            cp.ones(1 - odd),
            cp.zeros((n + odd) / 2 - 1),
        )
    )

    m = fft.ifft(cp.exp(cp.fft.fft(window * ceps))).real

    return m
