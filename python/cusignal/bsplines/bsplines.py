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

_gauss_spline_kernel = cp.ElementwiseKernel(
    "T x, float64 pi, float64 signsq, float64 r_siqnsq",
    "T output",
    """
    output = 1 / sqrt( 2 * pi * signsq ) * exp( -(x * x) * r_siqnsq );
    """,
    "_gauss_spline_kernel",
)


def gauss_spline(x, n):
    """Gaussian approximation to B-spline basis function of order n.

    Parameters
    ----------
    n : int
        The order of the spline. Must be nonnegative, i.e. n >= 0

    References
    ----------
    .. [1] Bouma H., Vilanova A., Bescos J.O., ter Haar Romeny B.M., Gerritsen
       F.A. (2007) Fast and Accurate Gaussian Derivatives Based on B-Splines.
       In: Sgallari F., Murli A., Paragios N. (eds) Scale Space and Variational
       Methods in Computer Vision. SSVM 2007. Lecture Notes in Computer
       Science, vol 4485. Springer, Berlin, Heidelberg
    """
    x = cp.asarray(x)

    signsq = (n + 1) / 12.0
    r_signsq = 0.5 / signsq
    return _gauss_spline_kernel(x, np.pi, signsq, r_signsq)


_cubic_kernel = cp.ElementwiseKernel(
    "T x",
    "T res",
    """
    T ax = abs( x );
    
    if( ax < 1 ) {
        res =  2.0 / 3 - 1.0 / 2  * ax * ax * ( 2 - ax );
    } else if( !( ax < 1 ) && ( ax < 2 ) ) {
        res = 1.0 / 6 * ( 2 - ax ) *  ( 2 - ax ) * ( 2 - ax );
    } else {
        res = 0;
    }
    """,
    "_cubic_kernel",
)


def cubic(x):
    """A cubic B-spline.

    This is a special case of `bspline`, and equivalent to ``bspline(x, 3)``.
    """
    x = cp.asarray(x)

    return _cubic_kernel(x)


_quadratic_kernel = cp.ElementwiseKernel(
    "T x",
    "T res",
    """
    T ax = abs( x );

    if( ax < 0.5 ) {
        res = 0.75 - ax * ax;
    } else if( !( ax < 0.5 ) && ( ax < 1.5 ) ) {
        res = ( ( ax - 1.5 ) * ( ax - 1.5 ) ) * 0.5 ;
    } else {
        res = 0;
    }
    """,
    "_quadratic_kernel",
)


def quadratic(x):
    """A quadratic B-spline.

    This is a special case of `bspline`, and equivalent to ``bspline(x, 2)``.
    """
    x = cp.asarray(x)

    return _quadratic_kernel(x)


def _coeff_smooth(lam):
    xi = 1 - 96 * lam + 24 * lam * cp.sqrt(3 + 144 * lam)
    omeg = cp.arctan2(cp.sqrt(144 * lam - 1), cp.sqrt(xi))
    rho = (24 * lam - 1 - cp.sqrt(xi)) / (24 * lam)
    rho = rho * cp.sqrt((48 * lam + 24 * lam * cp.sqrt(3 + 144 * lam)) / xi)
    return rho, omeg


def _hc(k, cs, rho, omega):
    return (
        cs
        / cp.sin(omega)
        * (rho ** k)
        * cp.sin(omega * (k + 1))
        * cp.greater(k, -1)
    )


def _hs(k, cs, rho, omega):
    c0 = (
        cs
        * cs
        * (1 + rho * rho)
        / (1 - rho * rho)
        / (1 - 2 * rho * rho * cp.cos(2 * omega) + rho ** 4)
    )
    gamma = (1 - rho * rho) / (1 + rho * rho) / cp.tan(omega)
    ak = abs(k)
    return c0 * rho ** ak * (cp.cos(omega * ak) + gamma * cp.sin(omega * ak))


def _cubic_smooth_coeff(signal, lamb):
    rho, omega = _coeff_smooth(lamb)
    cs = 1 - 2 * rho * cp.cos(omega) + rho * rho
    K = len(signal)
    yp = zeros((K,), signal.dtype.char)
    k = arange(K)
    yp[0] = _hc(0, cs, rho, omega) * signal[0] + add(
        _hc(k + 1, cs, rho, omega) * signal
    )

    yp[1] = (
        _hc(0, cs, rho, omega) * signal[0]
        + _hc(1, cs, rho, omega) * signal[1]
        + cp.add(_hc(k + 2, cs, rho, omega) * signal)
    )

    for n in range(2, K):
        yp[n] = (
            cs * signal[n]
            + 2 * rho * cp.cos(omega) * yp[n - 1]
            - rho * rho * yp[n - 2]
        )

    y = cp.zeros((K,), signal.dtype.char)

    y[K - 1] = cp.add(
        (_hs(k, cs, rho, omega) + _hs(k + 1, cs, rho, omega)) * signal[::-1]
    )
    y[K - 2] = cp.add(
        (_hs(k - 1, cs, rho, omega) + _hs(k + 2, cs, rho, omega))
        * signal[::-1]
    )

    for n in range(K - 3, -1, -1):
        y[n] = (
            cs * yp[n]
            + 2 * rho * cp.cos(omega) * y[n + 1]
            - rho * rho * y[n + 2]
        )

    return y


def _cubic_coeff(signal):
    zi = -2 + cp.sqrt(3)
    K = len(signal)
    yplus = cp.zeros((K,), signal.dtype.char)
    powers = zi ** cp.arange(K)
    yplus[0] = signal[0] + zi * cp.sum(powers * signal)
    for k in range(1, K):
        yplus[k] = signal[k] + zi * yplus[k - 1]
    output = cp.zeros((K,), signal.dtype)
    output[K - 1] = zi / (zi - 1) * yplus[K - 1]
    for k in range(K - 2, -1, -1):
        output[k] = zi * (output[k + 1] - yplus[k])
    return output * 6.0


# def _quadratic_coeff(signal):
#     zi = -3 + 2 * cp.sqrt(2.0)
#     K = len(signal)
#     yplus = cp.zeros((K,), signal.dtype.char)
#     powers = zi ** cp.arange(K)
#     yplus[0] = signal[0] + zi * cp.sum(powers * signal)
#     for k in range(1, K):
#         yplus[k] = signal[k] + zi * yplus[k - 1]
#     output = cp.zeros((K,), signal.dtype.char)
#     output[K - 1] = zi / (zi - 1) * yplus[K - 1]
#     for k in range(K - 2, -1, -1):
#         output[k] = zi * (output[k + 1] - yplus[k])
#     return output * 8.0


def cspline1d(signal, lamb=0.0):
    """
    Compute cubic spline coefficients for rank-1 array.

    Find the cubic spline coefficients for a 1-D signal assuming
    mirror-symmetric boundary conditions.   To obtain the signal back from the
    spline representation mirror-symmetric-convolve these coefficients with a
    length 3 FIR window [1.0, 4.0, 1.0]/ 6.0 .

    Parameters
    ----------
    signal : ndarray
        A rank-1 array representing samples of a signal.
    lamb : float, optional
        Smoothing coefficient, default is 0.0.

    Returns
    -------
    c : ndarray
        Cubic spline coefficients.

    """
    signal = cp.asarray(signal)
    if lamb != 0.0:
        return _cubic_smooth_coeff(signal, lamb)
    else:
        return _cubic_coeff(signal)
