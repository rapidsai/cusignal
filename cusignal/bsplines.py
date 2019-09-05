import cupy as cp
from cupy import (logical_and, asarray, pi, zeros_like,
                   array, arctan2, tan, zeros, arange, floor)
from cupy import (abs, sqrt, exp, greater, less, cos, add, sin,
                              less_equal, greater_equal)


def gauss_spline(x, n):
    """Gaussian approximation to B-spline basis function of order n.

    Parameters
    ----------
    n : int
        The order of the spline. Must be nonnegative, i.e. n >= 0

    References
    ----------
    .. [1] Bouma H., Vilanova A., Bescos J.O., ter Haar Romeny B.M., Gerritsen
       F.A. (2007) Fast and Accurate Gaussian Derivatives Based on B-Splines. In:
       Sgallari F., Murli A., Paragios N. (eds) Scale Space and Variational
       Methods in Computer Vision. SSVM 2007. Lecture Notes in Computer
       Science, vol 4485. Springer, Berlin, Heidelberg
   """
    x = asarray(x)
    signsq = (n + 1) / 12.0
    return 1 / sqrt(2 * pi * signsq) * exp(-x ** 2 / 2 / signsq)


def cubic(x):
    """A cubic B-spline.

    This is a special case of `bspline`, and equivalent to ``bspline(x, 3)``.
    """
    ax = abs(asarray(x))
    res = zeros_like(ax)
    cond1 = less(ax, 1)
    if cond1.any():
        ax1 = ax[cond1]
        res[cond1] = 2.0 / 3 - 1.0 / 2 * ax1 ** 2 * (2 - ax1)
    cond2 = ~cond1 & less(ax, 2)
    if cond2.any():
        ax2 = ax[cond2]
        res[cond2] = 1.0 / 6 * (2 - ax2) ** 3
    return res


def quadratic(x):
    """A quadratic B-spline.

    This is a special case of `bspline`, and equivalent to ``bspline(x, 2)``.
    """
    ax = abs(asarray(x))
    res = zeros_like(ax)
    cond1 = less(ax, 0.5)
    if cond1.any():
        ax1 = ax[cond1]
        res[cond1] = 0.75 - ax1 ** 2
    cond2 = ~cond1 & less(ax, 1.5)
    if cond2.any():
        ax2 = ax[cond2]
        res[cond2] = (ax2 - 1.5) ** 2 / 2.0
    return res


def _coeff_smooth(lam):
    xi = 1 - 96 * lam + 24 * lam * sqrt(3 + 144 * lam)
    omeg = arctan2(sqrt(144 * lam - 1), sqrt(xi))
    rho = (24 * lam - 1 - sqrt(xi)) / (24 * lam)
    rho = rho * sqrt((48 * lam + 24 * lam * sqrt(3 + 144 * lam)) / xi)
    return rho, omeg


def _hc(k, cs, rho, omega):
    return (cs / sin(omega) * (rho ** k) * sin(omega * (k + 1)) *
            greater(k, -1))


def _hs(k, cs, rho, omega):
    c0 = (cs * cs * (1 + rho * rho) / (1 - rho * rho) /
          (1 - 2 * rho * rho * cos(2 * omega) + rho ** 4))
    gamma = (1 - rho * rho) / (1 + rho * rho) / tan(omega)
    ak = abs(k)
    return c0 * rho ** ak * (cos(omega * ak) + gamma * sin(omega * ak))


def _cubic_smooth_coeff(signal, lamb):
    rho, omega = _coeff_smooth(lamb)
    cs = 1 - 2 * rho * cos(omega) + rho * rho
    K = len(signal)
    yp = zeros((K,), signal.dtype.char)
    k = arange(K)
    yp[0] = (_hc(0, cs, rho, omega) * signal[0] +
             add(_hc(k + 1, cs, rho, omega) * signal))

    yp[1] = (_hc(0, cs, rho, omega) * signal[0] +
             _hc(1, cs, rho, omega) * signal[1] +
             add(_hc(k + 2, cs, rho, omega) * signal))

    for n in range(2, K):
        yp[n] = (cs * signal[n] + 2 * rho * cos(omega) * yp[n - 1] -
                 rho * rho * yp[n - 2])

    y = zeros((K,), signal.dtype.char)

    y[K - 1] = add((_hs(k, cs, rho, omega) +
                           _hs(k + 1, cs, rho, omega)) * signal[::-1])
    y[K - 2] = add((_hs(k - 1, cs, rho, omega) +
                           _hs(k + 2, cs, rho, omega)) * signal[::-1])

    for n in range(K - 3, -1, -1):
        y[n] = (cs * yp[n] + 2 * rho * cos(omega) * y[n + 1] -
                rho * rho * y[n + 2])

    return y


def _cubic_coeff(signal):
    zi = -2 + sqrt(3)
    K = len(signal)
    yplus = zeros((K,), signal.dtype.char)
    powers = zi ** arange(K)
    yplus[0] = signal[0] + zi * cp.sum(powers * signal)
    for k in range(1, K):
        yplus[k] = signal[k] + zi * yplus[k - 1]
    output = zeros((K,), signal.dtype)
    output[K - 1] = zi / (zi - 1) * yplus[K - 1]
    for k in range(K - 2, -1, -1):
        output[k] = zi * (output[k + 1] - yplus[k])
    return output * 6.0


def _quadratic_coeff(signal):
    zi = -3 + 2 * sqrt(2.0)
    K = len(signal)
    yplus = zeros((K,), signal.dtype.char)
    powers = zi ** arange(K)
    yplus[0] = signal[0] + zi * cp.sum(powers * signal)
    for k in range(1, K):
        yplus[k] = signal[k] + zi * yplus[k - 1]
    output = zeros((K,), signal.dtype.char)
    output[K - 1] = zi / (zi - 1) * yplus[K - 1]
    for k in range(K - 2, -1, -1):
        output[k] = zi * (output[k + 1] - yplus[k])
    return output * 8.0


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
    signal = asarray(signal)
    if lamb != 0.0:
        return _cubic_smooth_coeff(signal, lamb)
    else:
        return _cubic_coeff(signal)