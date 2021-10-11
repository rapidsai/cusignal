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
from ..convolution.convolve import convolve
import numpy as np

_qmf_kernel = cp.ElementwiseKernel(
    "",
    "int64 output",
    """
    const int sign { ( i & 1 ) ? -1 : 1 };
    output = ( _ind.size() - ( i + 1 ) ) * sign;
    """,
    "_qmf_kernel",
    options=("-std=c++11",),
)


def qmf(hk):
    """
    Return high-pass qmf filter from low-pass

    Parameters
    ----------
    hk : array_like
        Coefficients of high-pass filter.

    """
    return _qmf_kernel(size=len(hk))


_morlet_kernel = cp.ElementwiseKernel(
    "float64 w, float64 s, bool complete",
    "complex128 output",
    """
    const double x { start + delta * i };

    thrust::complex<double> temp { exp(
        thrust::complex<double>( 0, w * x ) ) };

    if ( complete ) {
        temp -= exp( -0.5 * ( w * w ) );
    }

    output = temp * exp( -0.5 * ( x * x ) ) * pow( M_PI, -0.25 )
    """,
    "_morlet_kernel",
    options=("-std=c++11",),
    loop_prep="const double end { s * 2.0 * M_PI }; \
               const double start { -s * 2.0 * M_PI }; \
               const double delta { ( end - start ) / ( _ind.size() - 1 ) };",
)


def morlet(M, w=5.0, s=1.0, complete=True):
    """
    Complex Morlet wavelet.

    Parameters
    ----------
    M : int
        Length of the wavelet.
    w : float, optional
        Omega0. Default is 5
    s : float, optional
        Scaling factor, windowed from ``-s*2*pi`` to ``+s*2*pi``. Default is 1.
    complete : bool, optional
        Whether to use the complete or the standard version.

    Returns
    -------
    morlet : (M,) ndarray

    See Also
    --------
    cusignal.gausspulse

    Notes
    -----
    The standard version::

        pi**-0.25 * exp(1j*w*x) * exp(-0.5*(x**2))

    This commonly used wavelet is often referred to simply as the
    Morlet wavelet.  Note that this simplified version can cause
    admissibility problems at low values of `w`.

    The complete version::

        pi**-0.25 * (exp(1j*w*x) - exp(-0.5*(w**2))) * exp(-0.5*(x**2))

    This version has a correction
    term to improve admissibility. For `w` greater than 5, the
    correction term is negligible.

    Note that the energy of the return wavelet is not normalised
    according to `s`.

    The fundamental frequency of this wavelet in Hz is given
    by ``f = 2*s*w*r / M`` where `r` is the sampling rate.

    Note: This function was created before `cwt` and is not compatible
    with it.

    """
    return _morlet_kernel(w, s, complete, size=M)


_ricker_kernel = cp.ElementwiseKernel(
    "float64 a",
    "float64 total",
    """
    const double vec { i - ( _ind.size() - 1.0 ) * 0.5 };
    const double xsq { vec * vec };
    const double mod { 1 - xsq / wsq };
    const double gauss { exp( -xsq / ( 2.0 * wsq ) ) };

    total = A * mod * gauss;
    """,
    "_ricker_kernel",
    options=("-std=c++11",),
    loop_prep="const double A { 2.0 / ( sqrt( 3 * a ) * pow( M_PI, 0.25 ) ) }; \
               const double wsq { a * a };",
)


def ricker(points, a):
    """
    Return a Ricker wavelet, also known as the "Mexican hat wavelet".

    It models the function:

        ``A (1 - x^2/a^2) exp(-x^2/2 a^2)``,

    where ``A = 2/sqrt(3a)pi^1/4``.

    Parameters
    ----------
    points : int
        Number of points in `vector`.
        Will be centered around 0.
    a : scalar
        Width parameter of the wavelet.

    Returns
    -------
    vector : (N,) ndarray
        Array of length `points` in shape of ricker curve.

    Examples
    --------
    >>> import cusignal
    >>> import cupy as cp
    >>> import matplotlib.pyplot as plt

    >>> points = 100
    >>> a = 4.0
    >>> vec2 = cusignal.ricker(points, a)
    >>> print(len(vec2))
    100
    >>> plt.plot(cp.asnumpy(vec2))
    >>> plt.show()

    """
    return _ricker_kernel(a, size=points)


_morlet2_kernel = cp.ElementwiseKernel(
    "float64 w, float64 s",
    "complex128 output",
    """
    const double x { ( i - ( _ind.size() - 1.0 ) * 0.5 ) / s };

    thrust::complex<double> temp { exp(
        thrust::complex<double>( 0, w * x ) ) };

    output = sqrt( 1 / s ) * temp * exp( -0.5 * ( x * x ) ) *
        pow( M_PI, -0.25 )
    """,
    "_morlet_kernel",
    options=("-std=c++11",),
    loop_prep="",
)


def morlet2(M, s, w=5):
    """
    Complex Morlet wavelet, designed to work with `cwt`.
    Returns the complete version of morlet wavelet, normalised
    according to `s`::
        exp(1j*w*x/s) * exp(-0.5*(x/s)**2) * pi**(-0.25) * sqrt(1/s)
    Parameters
    ----------
    M : int
        Length of the wavelet.
    s : float
        Width parameter of the wavelet.
    w : float, optional
        Omega0. Default is 5
    Returns
    -------
    morlet : (M,) ndarray
    See Also
    --------
    morlet : Implementation of Morlet wavelet, incompatible with `cwt`
    Notes
    -----
    .. versionadded:: 1.4.0
    This function was designed to work with `cwt`. Because `morlet2`
    returns an array of complex numbers, the `dtype` argument of `cwt`
    should be set to `complex128` for best results.
    Note the difference in implementation with `morlet`.
    The fundamental frequency of this wavelet in Hz is given by::
        f = w*fs / (2*s*np.pi)
    where ``fs`` is the sampling rate and `s` is the wavelet width parameter.
    Similarly we can get the wavelet width parameter at ``f``::
        s = w*fs / (2*f*np.pi)
    Examples
    --------
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> M = 100
    >>> s = 4.0
    >>> w = 2.0
    >>> wavelet = signal.morlet2(M, s, w)
    >>> plt.plot(abs(wavelet))
    >>> plt.show()
    This example shows basic use of `morlet2` with `cwt` in time-frequency
    analysis:
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> t, dt = np.linspace(0, 1, 200, retstep=True)
    >>> fs = 1/dt
    >>> w = 6.
    >>> sig = np.cos(2*np.pi*(50 + 10*t)*t) + np.sin(40*np.pi*t)
    >>> freq = np.linspace(1, fs/2, 100)
    >>> widths = w*fs / (2*freq*np.pi)
    >>> cwtm = signal.cwt(sig, signal.morlet2, widths, w=w)
    >>> plt.pcolormesh(t, freq, np.abs(cwtm),
        cmap='viridis', shading='gouraud')
    >>> plt.show()
    """

    return _morlet2_kernel(w, s, size=M)


def cwt(data, wavelet, widths):
    """
    Continuous wavelet transform.

    Performs a continuous wavelet transform on `data`,
    using the `wavelet` function. A CWT performs a convolution
    with `data` using the `wavelet` function, which is characterized
    by a width parameter and length parameter.

    Parameters
    ----------
    data : (N,) ndarray
        data on which to perform the transform.
    wavelet : function
        Wavelet function, which should take 2 arguments.
        The first argument is the number of points that the returned vector
        will have (len(wavelet(length,width)) == length).
        The second is a width parameter, defining the size of the wavelet
        (e.g. standard deviation of a gaussian). See `ricker`, which
        satisfies these requirements.
    widths : (M,) sequence
        Widths to use for transform.

    Returns
    -------
    cwt: (M, N) ndarray
        Will have shape of (len(widths), len(data)).

    Notes
    -----
    ::

        length = min(10 * width[ii], len(data))
        cwt[ii,:] = cusignal.convolve(data, wavelet(length,
                                    width[ii]), mode='same')

    Examples
    --------
    >>> import cusignal
    >>> import cupy as cp
    >>> import matplotlib.pyplot as plt
    >>> t = cp.linspace(-1, 1, 200, endpoint=False)
    >>> sig  = cp.cos(2 * cp.pi * 7 * t) + cusignal.gausspulse(t - 0.4, fc=2)
    >>> widths = cp.arange(1, 31)
    >>> cwtmatr = cusignal.cwt(sig, cusignal.ricker, widths)
    >>> plt.imshow(abs(cp.asnumpy(cwtmatr)), extent=[-1, 1, 31, 1],
                   cmap='PRGn', aspect='auto', vmax=abs(cwtmatr).max(),
                   vmin=-abs(cwtmatr).max())
    >>> plt.show()

    """
    if cp.asarray(wavelet(1, 1)).dtype.char in 'FDG':
        dtype = cp.complex128
    else:
        dtype = cp.float64

    output = cp.empty([len(widths), len(data)], dtype=dtype)

    for ind, width in enumerate(widths):
        N = np.min([10 * int(width), len(data)])
        wavelet_data = cp.conj(wavelet(N, int(width)))[::-1]
        output[ind, :] = convolve(data, wavelet_data, mode="same")
    return output
