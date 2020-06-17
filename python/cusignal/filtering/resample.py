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
from cupy import (
    arange,
    asarray,
    ndarray,
    zeros,
)
from cupyx.scipy import fftpack
from cupy.fft import ifftshift

from math import gcd

from ..windows.windows import get_window
from ._upfirdn_cuda import _UpFIRDn, _output_len
from ..filter_design.fir_filter_design import firwin


def _design_resample_poly(up, down, window):
    """
    Design a prototype FIR low-pass filter using the window method
    for use in polyphase rational resampling.

    Parameters
    ----------
    up : int
        The upsampling factor.
    down : int
        The downsampling factor.
    window : string or tuple
        Desired window to use to design the low-pass filter.
        See below for details.

    Returns
    -------
    h : array
        The computed FIR filter coefficients.

    See Also
    --------
    resample_poly : Resample up or down using the polyphase method.

    Notes
    -----
    The argument `window` specifies the FIR low-pass filter design.
    The functions `cusignal.get_window` and `cusignal.firwin`
    are called to generate the appropriate filter coefficients.

    The returned array of coefficients will always be of data type
    `complex128` to maintain precision. For use in lower-precision
    filter operations, this array should be converted to the desired
    data type before providing it to `cusignal.resample_poly`.

    """

    # Determine our up and down factors
    # Use a rational approximation to save computation time on really long
    # signals
    g_ = gcd(up, down)
    up //= g_
    down //= g_

    # Design a linear-phase low-pass FIR filter
    max_rate = max(up, down)
    f_c = 1.0 / max_rate  # cutoff of FIR filter (rel. to Nyquist)

    # reasonable cutoff for our sinc-like function
    half_len = 10 * max_rate

    h = firwin(2 * half_len + 1, f_c, window=window)
    return h


def decimate(
    x, q, n=None, axis=-1, zero_phase=True,
):
    """
    Downsample the signal after applying an anti-aliasing filter.
    Parameters
    ----------
    x : array_like
        The signal to be downsampled, as an N-dimensional array.
    q : int
        The downsampling factor.
    n : int or array_like, optional
        The order of the filter (1 less than the length for FIR) to calculate,
        or the FIR filter coefficients to employ. Defaults to calculating the
        coefficients for 20 times the downsampling factor.
    axis : int, optional
        The axis along which to decimate.
    zero_phase : bool, optional
        Prevent shifting the outputs back by the filter's
        group delay when using an FIR filter. The default value of ``True`` is
        recommended, since a phase shift is generally not desired.

    Returns
    -------
    y : ndarray
        The down-sampled signal.
    See Also
    --------
    resample : Resample up or down using the FFT method.
    resample_poly : Resample using polyphase filtering and an FIR filter.
    Notes
    -----
    Only FIR filter types are currently supported in cuSignal.
    """

    x = asarray(x)
    if isinstance(n, (list, ndarray)):
        b = asarray(n)
    else:
        if n is None:
            half_len = 10 * q  # reasonable cutoff for our sinc-like function
            n = 2 * half_len

        b = firwin(n + 1, 1.0 / q, window="hamming")

    sl = [slice(None)] * x.ndim

    if zero_phase:
        y = resample_poly(x, 1, q, axis=axis, window=b)
    else:
        # upfirdn is generally faster than lfilter by a factor equal to the
        # downsampling factor, since it only calculates the needed outputs
        n_out = x.shape[axis] // q + bool(x.shape[axis] % q)
        y = upfirdn(b, x, 1, q, axis)
        sl[axis] = slice(None, n_out, None)

    return y[tuple(sl)]


def resample(x, num, t=None, axis=0, window=None, domain="time"):
    """
    Resample `x` to `num` samples using Fourier method along the given axis.

    The resampled signal starts at the same value as `x` but is sampled
    with a spacing of ``len(x) / num * (spacing of x)``.  Because a
    Fourier method is used, the signal is assumed to be periodic.

    Parameters
    ----------
    x : array_like
        The data to be resampled.
    num : int
        The number of samples in the resampled signal.
    t : array_like, optional
        If `t` is given, it is assumed to be the sample positions
        associated with the signal data in `x`.
    axis : int, optional
        The axis of `x` that is resampled.  Default is 0.
    window : array_like, callable, string, float, or tuple, optional
        Specifies the window applied to the signal in the Fourier
        domain.  See below for details.
    domain : string, optional
        A string indicating the domain of the input `x`:

        ``time``
           Consider the input `x` as time-domain. (Default)
        ``freq``
           Consider the input `x` as frequency-domain.

    Returns
    -------
    resampled_x or (resampled_x, resampled_t)
        Either the resampled array, or, if `t` was given, a tuple
        containing the resampled array and the corresponding resampled
        positions.

    See Also
    --------
    decimate : Downsample the signal after applying an FIR or IIR filter.
    resample_poly : Resample using polyphase filtering and an FIR filter.

    Notes
    -----
    The argument `window` controls a Fourier-domain window that tapers
    the Fourier spectrum before zero-padding to alleviate ringing in
    the resampled values for sampled signals you didn't intend to be
    interpreted as band-limited.

    If `window` is a function, then it is called with a vector of inputs
    indicating the frequency bins (i.e. fftfreq(x.shape[axis]) ).

    If `window` is an array of the same length as `x.shape[axis]` it is
    assumed to be the window to be applied directly in the Fourier
    domain (with dc and low-frequency first).

    For any other type of `window`, the function `cusignal.get_window`
    is called to generate the window.

    The first sample of the returned vector is the same as the first
    sample of the input vector.  The spacing between samples is changed
    from ``dx`` to ``dx * len(x) / num``.

    If `t` is not None, then it represents the old sample positions,
    and the new sample positions will be returned as well as the new
    samples.

    As noted, `resample` uses FFT transformations, which can be very
    slow if the number of input or output samples is large and prime;
    see `scipy.fftpack.fft`.

    Examples
    --------
    Note that the end of the resampled data rises to meet the first
    sample of the next cycle:

    >>> import cusignal
    >>> import cupy as cp

    >>> x = cp.linspace(0, 10, 20, endpoint=False)
    >>> y = cp.cos(-x**2/6.0)
    >>> f = cusignal.resample(y, 100)
    >>> xnew = cp.linspace(0, 10, 100, endpoint=False)

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(x, y, 'go-', xnew, f, '.-', 10, y[0], 'ro')
    >>> plt.legend(['data', 'resampled'], loc='best')
    >>> plt.show()
    """
    x = asarray(x)
    Nx = x.shape[axis]

    if domain == "time":
        X = fftpack.fft(x, axis=axis)
    elif domain == "freq":
        X = x
    else:
        raise NotImplementedError("domain should be 'time' or 'freq'")

    if window is not None:
        if callable(window):
            W = window(fftpack.fftfreq(Nx))
        elif isinstance(window, ndarray):
            if window.shape != (Nx,):
                raise ValueError("window must have the same length as data")
            W = window
        else:
            W = ifftshift(get_window(window, Nx))
        newshape = [1] * x.ndim
        newshape[axis] = len(W)
        W.shape = newshape
        X = X * W
    sl = [slice(None)] * x.ndim
    newshape = list(x.shape)
    newshape[axis] = num
    N = int(cp.minimum(num, Nx))
    Y = zeros(newshape, "D")
    sl[axis] = slice(0, (N + 1) // 2)
    Y[sl] = X[sl]
    sl[axis] = slice(-(N - 1) // 2, None)
    Y[sl] = X[sl]
    y = fftpack.ifft(Y, axis=axis) * (float(num) / float(Nx))

    if x.dtype.char not in ["F", "D"]:
        y = y.real

    if t is None:
        return y
    else:
        new_t = arange(0, num) * (t[1] - t[0]) * Nx / float(num) + t[0]
        return y, new_t


def resample_poly(
    x, up, down, axis=0, window=("kaiser", 5.0),
):
    """
    Resample `x` along the given axis using polyphase filtering.

    The signal `x` is upsampled by the factor `up`, a zero-phase low-pass
    FIR filter is applied, and then it is downsampled by the factor `down`.
    The resulting sample rate is ``up / down`` times the original sample
    rate. Values beyond the boundary of the signal are assumed to be zero
    during the filtering step.

    Parameters
    ----------
    x : array_like
        The data to be resampled.
    up : int
        The upsampling factor.
    down : int
        The downsampling factor.
    axis : int, optional
        The axis of `x` that is resampled. Default is 0.
    window : string, tuple, or array_like, optional
        Desired window to use to design the low-pass filter, or the FIR filter
        coefficients to employ. See below for details.

    Returns
    -------
    resampled_x : array
        The resampled array.

    See Also
    --------
    decimate : Downsample the signal after applying an FIR or IIR filter.
    resample : Resample up or down using the FFT method.

    Notes
    -----
    This polyphase method will likely be faster than the Fourier method
    in `cusignal.resample` when the number of samples is large and
    prime, or when the number of samples is large and `up` and `down`
    share a large greatest common denominator. The length of the FIR
    filter used will depend on ``max(up, down) // gcd(up, down)``, and
    the number of operations during polyphase filtering will depend on
    the filter length and `down` (see `cusignal.upfirdn` for details).

    The argument `window` specifies the FIR low-pass filter design.

    If `window` is an array_like it is assumed to be the FIR filter
    coefficients. Note that the FIR filter is applied after the upsampling
    step, so it should be designed to operate on a signal at a sampling
    frequency higher than the original by a factor of `up//gcd(up, down)`.
    This function's output will be centered with respect to this array, so it
    is best to pass a symmetric filter with an odd number of samples if, as
    is usually the case, a zero-phase filter is desired.

    For any other type of `window`, the functions `cusignal.get_window`
    and `cusignal.firwin` are called to generate the appropriate filter
    coefficients.

    The first sample of the returned vector is the same as the first
    sample of the input vector. The spacing between samples is changed
    from ``dx`` to ``dx * down / float(up)``.

    Examples
    --------
    Note that the end of the resampled data rises to meet the first
    sample of the next cycle for the FFT method, and gets closer to zero
    for the polyphase method:

    >>> from scipy import signal

    >>> x = cp.linspace(0, 10, 20, endpoint=False)
    >>> y = cp.cos(-x**2/6.0)
    >>> f_fft = cusignal.resample(y, 100)
    >>> f_poly = cusignal.resample_poly(y, 100, 20)
    >>> xnew = cp.linspace(0, 10, 100, endpoint=False)

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(xnew, f_fft, 'b.-', xnew, f_poly, 'r.-')
    >>> plt.plot(x, y, 'ko-')
    >>> plt.plot(10, y[0], 'bo', 10, 0., 'ro')  # boundaries
    >>> plt.legend(['resample', 'resamp_poly', 'data'], loc='best')
    >>> plt.show()
    """

    x = asarray(x)
    up = int(up)
    down = int(down)
    if up < 1 or down < 1:
        raise ValueError("up and down must be >= 1")

    # Determine our up and down factors
    # Use a rational approimation to save computation time on really long
    # signals
    g_ = gcd(up, down)
    up //= g_
    down //= g_
    if up == down == 1:
        return x.copy()
    n_out = x.shape[axis] * up
    n_out = n_out // down + bool(n_out % down)

    if isinstance(window, (list, ndarray)):
        window = asarray(window)
        if window.ndim > 1:
            raise ValueError("window must be 1-D")
        half_len = (window.size - 1) // 2
        h = up * window
    else:
        half_len = 10 * max(up, down)
        h = up * _design_resample_poly(up, down, window)

    # Zero-pad our filter to put the output samples at the center
    n_pre_pad = down - half_len % down
    n_post_pad = 0
    n_pre_remove = (half_len + n_pre_pad) // down
    # We should rarely need to do this given our filter lengths...
    while (
        _output_len(len(h) + n_pre_pad + n_post_pad, x.shape[axis], up, down)
        < n_out + n_pre_remove
    ):
        n_post_pad += 1

    h = cp.concatenate(
        (zeros(n_pre_pad, h.dtype), h, zeros(n_post_pad, h.dtype))
    )
    n_pre_remove_end = n_pre_remove + n_out

    # filter then remove excess
    y = upfirdn(h, x, up, down, axis)
    keep = [slice(None)] * x.ndim
    keep[axis] = slice(n_pre_remove, n_pre_remove_end)

    return y[tuple(keep)]


def upfirdn(
    h, x, up=1, down=1, axis=-1,
):
    """
    Upsample, FIR filter, and downsample

    Parameters
    ----------
    h : array_like
        1-dimensional FIR (finite-impulse response) filter coefficients.
    x : array_like
        Input signal array.
    up : int, optional
        Upsampling rate. Default is 1.
    down : int, optional
        Downsampling rate. Default is 1.
    axis : int, optional
        The axis of the input data array along which to apply the
        linear filter. The filter is applied to each subarray along
        this axis. Default is -1.

    Returns
    -------
    y : ndarray
        The output signal array. Dimensions will be the same as `x` except
        for along `axis`, which will change size according to the `h`,
        `up`,  and `down` parameters.

    Notes
    -----
    The algorithm is an implementation of the block diagram shown on page 129
    of the Vaidyanathan text [1]_ (Figure 4.3-8d).
    .. [1] P. P. Vaidyanathan, Multirate Systems and Filter Banks,
       Prentice Hall, 1993.
    The direct approach of upsampling by factor of P with zero insertion,
    FIR filtering of length ``N``, and downsampling by factor of Q is
    O(N*Q) per output sample. The polyphase implementation used here is
    O(N/P).

    Examples
    --------
    Simple operations:
    >>> from cusignal import upfirdn
    >>> upfirdn([1, 1, 1], [1, 1, 1])   # FIR filter
    array([ 1.,  2.,  3.,  2.,  1.])
    >>> upfirdn([1], [1, 2, 3], 3)  # upsampling with zeros insertion
    array([ 1.,  0.,  0.,  2.,  0.,  0.,  3.,  0.,  0.])
    >>> upfirdn([1, 1, 1], [1, 2, 3], 3)  # upsampling with sample-and-hold
    array([ 1.,  1.,  1.,  2.,  2.,  2.,  3.,  3.,  3.])
    >>> upfirdn([.5, 1, .5], [1, 1, 1], 2)  # linear interpolation
    array([ 0.5,  1. ,  1. ,  1. ,  1. ,  1. ,  0.5,  0. ])
    >>> upfirdn([1], cp.arange(10), 1, 3)  # decimation by 3
    array([ 0.,  3.,  6.,  9.])
    >>> upfirdn([.5, 1, .5], cp.arange(10), 2, 3)  # linear interp, rate 2/3
    array([ 0. ,  1. ,  2.5,  4. ,  5.5,  7. ,  8.5,  0. ])
    Apply a single filter to multiple signals:
    >>> x = cp.reshape(cp.arange(8), (4, 2))
    >>> x
    array([[0, 1],
           [2, 3],
           [4, 5],
           [6, 7]])
    Apply along the last dimension of ``x``:
    >>> h = [1, 1]
    >>> upfirdn(h, x, 2)
    array([[ 0.,  0.,  1.,  1.],
           [ 2.,  2.,  3.,  3.],
           [ 4.,  4.,  5.,  5.],
           [ 6.,  6.,  7.,  7.]])
    Apply along the 0th dimension of ``x``:
    >>> upfirdn(h, x, 2, axis=0)
    array([[ 0.,  1.],
           [ 0.,  1.],
           [ 2.,  3.],
           [ 2.,  3.],
           [ 4.,  5.],
           [ 4.,  5.],
           [ 6.,  7.],
           [ 6.,  7.]])
    """

    x = cp.asarray(x)
    ufd = _UpFIRDn(h, x.dtype, up, down)
    # This is equivalent to (but faster than) using cp.apply_along_axis
    return ufd.apply_filter(x, axis)
