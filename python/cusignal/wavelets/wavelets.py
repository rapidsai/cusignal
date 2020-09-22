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
from ..convolution.convolve import convolve

_qmf_kernel = cp.ElementwiseKernel(
    "T N",
    "T output",
    """
    int sign = ( i % 2 ) ? -1 : 1;
    output = (N - (i + 1)) * sign;
    """,
    "_qmf_kernel",
)


def qmf(hk):
    """
    Return high-pass qmf filter from low-pass

    Parameters
    ----------
    hk : array_like
        Coefficients of high-pass filter.

    """
    N = len(hk)
    output = cp.empty(N, dtype=cp.int64)

    _qmf_kernel(N, output)

    return output


_morlet_kernel = cp.ElementwiseKernel(
    "float64 delta, float64 start, float64 w, float64 pi, bool complete",
    "T output",
    """
    double x = start + delta * i;

    T temp = T(0, w * x);

    temp = exp(temp);

    if (complete) {
        temp -= exp( -0.5 * (w * w) );
    }

    output = temp * exp( -0.5 * (x * x)) * pow(pi, -0.25)
    """,
    "_morlet_kernel",
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
    output = cp.empty(M, dtype=cp.complex128)
    end = s * 2 * np.pi
    start = -s * 2 * np.pi
    delta = (end - start) / (M - 1)

    _morlet_kernel(delta, start, w, np.pi, complete, output, size=M)

    return output


_ricker_kernel = cp.ElementwiseKernel(
    "T A, T wsq",
    "T total",
    """
    T vec = i - (_ind.size() - 1.0) / 2;
    T xsq = vec * vec;
    T mod = ( 1 - xsq / wsq );
    T gauss = exp( -xsq / ( 2 * wsq ) );
    total = A * mod * gauss;
    """,
    "_ricker_kernel",
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
    total = cp.empty(int(points), dtype=cp.float64)

    A = 2 / (np.sqrt(3 * a) * (np.pi ** 0.25))
    wsq = a ** 2

    _ricker_kernel(float(A), float(wsq), total, size=points)

    return total


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
    >>> plt.imshow(cp.asnumpy(cwtmatr), extent=[-1, 1, 31, 1], cmap='PRGn',
                   aspect='auto', vmax=abs(cwtmatr).max(),
                   vmin=-abs(cwtmatr).max())
    >>> plt.show()

    """
    output = cp.empty([len(widths), len(data)])
    for ind, width in enumerate(widths):
        wavelet_data = wavelet(min(10 * width, len(data)), width)
        output[ind, :] = convolve(data, wavelet_data, mode="same")
    return output
