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


_real_cepstrum_kernel = cp.ElementwiseKernel(
    "T spectrum",
    "T output",
    """
    output = log( abs( spectrum ) );
    """,
    "_real_cepstrum_kernel",
    options=("-std=c++11",),
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
    spectrum = cp.fft.fft(x, n=n, axis=axis)
    spectrum = _real_cepstrum_kernel(spectrum)
    return cp.fft.ifft(spectrum, n=n, axis=axis).real


_complex_cepstrum_kernel = cp.ElementwiseKernel(
    "C spectrum, raw T unwrapped",
    "C output, T ndelay",
    """
    ndelay = round( unwrapped[center] / M_PI );
    const T temp { unwrapped[i] - ( M_PI * ndelay * i / center ) };

    output = log( abs( spectrum ) ) + C( 0, temp );
    """,
    "_complex_cepstrum_kernel",
    options=("-std=c++11",),
    return_tuple=True,
    loop_prep="const int center { static_cast<int>( 0.5 * \
        ( _ind.size() + 1 ) ) };",
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
    spectrum = cp.fft.fft(x, n=n, axis=axis)
    unwrapped = cp.unwrap(cp.angle(spectrum))
    log_spectrum, ndelay = _complex_cepstrum_kernel(spectrum, unwrapped)
    ceps = cp.fft.ifft(log_spectrum, n=n, axis=axis).real

    return ceps, ndelay


_inverse_complex_cepstrum_kernel = cp.ElementwiseKernel(
    "C log_spectrum, int32 ndelay, float64 pi",
    "C spectrum",
    """
    const double wrapped { log_spectrum.imag() + M_PI * ndelay * i / center };

    spectrum = exp( C( log_spectrum.real(), wrapped ) )
    """,
    "_inverse_complex_cepstrum_kernel",
    options=("-std=c++11",),
    loop_prep="const double center { 0.5 * ( _ind.size() + 1 ) };",
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
    ceps = cp.asarray(ceps)
    log_spectrum = cp.fft.fft(ceps)
    spectrum = _inverse_complex_cepstrum_kernel(log_spectrum, ndelay, cp.pi)
    iceps = cp.fft.ifft(spectrum).real

    return iceps


_minimum_phase_kernel = cp.ElementwiseKernel(
    "T ceps",
    "T window",
    """
    if ( !i ) {
        window = ceps;
    } else if ( i < bend ) {
        window = ceps * 2.0;
    } else if ( i == bend ) {
        window = ceps * ( 1 - odd );
    } else {
        window = 0;
    }
    """,
    "_minimum_phase_kernel",
    options=("-std=c++11",),
    loop_prep="const bool odd { _ind.size() & 1 }; \
               const int bend { static_cast<int>( 0.5 * \
                    ( _ind.size() + odd ) ) };",
)


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
    window = _minimum_phase_kernel(ceps)
    m = cp.fft.ifft(cp.exp(cp.fft.fft(window))).real

    return m
