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

    h = fft.fft(x, n=n, axis=axis)
    ceps = fft.ifft(cp.log(cp.abs(h)), n=n, axis=axis).real

    return ceps


def cceps_unwrap(x):
    r"""
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

    h = fft.fft(x, n=n, axis=axis)
    ah = cceps_unwrap(cp.angle(h))
    logh = cp.log(cp.abs(h)) + 1j * ah
    cceps = fft.ifft(logh, n=n, axis=axis).real

    return cceps
