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
from cupy import linalg

import numpy as np

from ._channelizer_cuda import _channelizer
from ..convolution.correlate import correlate
from ..filter_design.filter_design_utils import _validate_sos
from ._sosfilt_cuda import _sosfilt
from ..convolution.convolve import fftconvolve
from ..utils.helper_tools import _get_max_smem, _get_max_tpb


_wiener_prep_kernel = cp.ElementwiseKernel(
    "T iMean, T iVar, T prod",
    "T lMean, T lVar",
    """
    // Estimate the local mean
    T temp { iMean / prod };
    lMean = temp;
    lVar = iVar / prod - (temp * temp);
    """,
    "_wiener_prep_kernel",
    options=("-std=c++11",),
)

_wiener_post_kernel = cp.ElementwiseKernel(
    "T im, T lMean, T lVar, T noise",
    "T out",
    """
    // Estimate the local mean
    T res { im - lMean };
    res *= 1.0 - noise / lVar;
    res += lMean;
    if ( lVar < noise ) {
        out = lMean;
    } else {
        out = res;
    }
    """,
    "_wiener_post_kernel",
    options=("-std=c++11",),
)


def wiener(im, mysize=None, noise=None):
    """
    Perform a Wiener filter on an N-dimensional array.

    Apply a Wiener filter to the N-dimensional array `im`.

    Parameters
    ----------
    im : ndarray
        An N-dimensional array.
    mysize : int or array_like, optional
        A scalar or an N-length list giving the size of the Wiener filter
        window in each dimension.  Elements of mysize should be odd.
        If mysize is a scalar, then this scalar is used as the size
        in each dimension.
    noise : float, optional
        The noise-power to use. If None, then noise is estimated as the
        average of the local variance of the input.

    Returns
    -------
    out : ndarray
        Wiener filtered result with the same shape as `im`.

    """
    im = cp.asarray(im)
    if mysize is None:
        mysize = [3] * im.ndim
    mysize = np.asarray(mysize)
    if mysize.shape == ():
        mysize = cp.repeat(mysize.item(), im.ndim)
        mysize = np.asarray(mysize)

    lprod = cp.prod(mysize, axis=0)
    lMean = correlate(im, cp.ones(mysize), "same")
    lVar = correlate(im ** 2, cp.ones(mysize), "same")

    lMean, lVar = _wiener_prep_kernel(lMean, lVar, lprod)

    # Estimate the noise power if needed.
    if noise is None:
        noise = cp.mean(cp.ravel(lVar), axis=0)

    return _wiener_post_kernel(im, lMean, lVar, noise)


def firfilter(b, x, axis=None, zi=None):
    """
    Filter data along one-dimension with an FIR filter.

    Filter a data sequence, `x`, using a digital filter. This works for many
    fundamental data types (including Object type). Please note, cuSignal
    doesn't support IIR filters presently, and this implementation is optimized
    for large filtering operations (and inherently depends on fftconvolve)

    Parameters
    ----------
    b : array_like
        The numerator coefficient vector in a 1-D sequence.
    x : array_like
        An N-dimensional input array.
    axis : int, optional
        The axis of the input data array along which to apply the
        linear filter. The filter is applied to each subarray along
        this axis.  Default is -1.
    zi : array_like, optional
        Initial conditions for the filter delays.  It is a vector
        (or array of vectors for an N-dimensional input) of length
        ``max(len(a), len(b)) - 1``.  If `zi` is None or is not given then
        initial rest is assumed.  See `lfiltic` for more information.

    Returns
    -------
    y : array
        The output of the digital filter.
    zf : array, optional
        If `zi` is None, this is not returned, otherwise, `zf` holds the
        final filter delay values.

    Examples
    -------
    >>> from scipy import signal
    >>> import cupy as cp
    >>> import cusignal
    >>> [b, a] = signal.butter(3, 0.5)
    >>> b = cp.asarray(b)
    >>> x = cp.random.randn(2**8)
    >>> y = cusignal.firfilter(b, x)

    """

    if zi is not None:
        raise NotImplementedError(
            "Initial conditions are not currently \
            supported"
        )

    y = fftconvolve(b, x, mode="full", axes=axis)
    return y[: len(x)]


def sosfilt(
    sos,
    x,
    axis=-1,
    zi=None,
):
    """
    Filter data along one dimension using cascaded second-order sections.
    Filter a data sequence, `x`, using a digital IIR filter defined by
    `sos`.

    Parameters
    ----------
    sos : array_like
        Array of second-order filter coefficients, must have shape
        ``(n_sections, 6)``. Each row corresponds to a second-order
        section, with the first three columns providing the numerator
        coefficients and the last three providing the denominator
        coefficients.
    x : array_like
        An N-dimensional input array.
    axis : int, optional
        The axis of the input data array along which to apply the
        linear filter. The filter is applied to each subarray along
        this axis.  Default is -1.
    zi : array_like, optional
        Initial conditions for the cascaded filter delays.  It is a (at
        least 2D) vector of shape ``(n_sections, ..., 2, ...)``, where
        ``..., 2, ...`` denotes the shape of `x`, but with ``x.shape[axis]``
        replaced by 2.  If `zi` is None or is not given then initial rest
        (i.e. all zeros) is assumed.
        Note that these initial conditions are *not* the same as the initial
        conditions given by `lfiltic` or `lfilter_zi`.

    Returns
    -------
    y : ndarray
        The output of the digital filter.
    zf : ndarray, optional
        If `zi` is None, this is not returned, otherwise, `zf` holds the
        final filter delay values.
    See Also
    --------
    zpk2sos, sos2zpk, sosfilt_zi, sosfiltfilt, sosfreqz

    Notes
    -----
    WARNING: This is an experimental API and is prone to change in future
    versions of cuSignal.

    The filter function is implemented as a series of second-order filters
    with direct-form II transposed structure. It is designed to minimize
    numerical precision errors for high-order filters.

    Limitations
    -----------
    1. The number of n_sections must be less than 513.
    2. The number of samples must be greater than the number of sections

    Examples
    --------
    sosfilt is a stable alternative to `lfilter` as using 2nd order sections
    reduces numerical error. We are working on building out sos filter output,
    so please submit GitHub feature requests as needed. You can also generate
    a filter on CPU with scipy.signal and then move that to GPU for actual
    filtering operations with `cp.asarray`.

    Plot a 13th-order filter's impulse response using both `sosfilt`:
    >>> from scipy import signal
    >>> import cusignal
    >>> import cupy as cp
    >>> # Generate filter on CPU with Scipy.Signal
    >>> sos = signal.ellip(13, 0.009, 80, 0.05, output='sos')
    >>> # Move data to GPU
    >>> sos = cp.asarray(sos)
    >>> x = cp.random.randn(100_000_000)
    >>> y = cusignal.sosfilt(sos, x)
    """

    x = cp.asarray(x)
    if x.ndim == 0:
        raise ValueError("x must be at least 1D")

    sos, n_sections = _validate_sos(sos)
    sos = cp.asarray(sos)

    x_zi_shape = list(x.shape)
    x_zi_shape[axis] = 2
    x_zi_shape = tuple([n_sections] + x_zi_shape)
    inputs = [sos, x]

    if zi is not None:
        inputs.append(np.asarray(zi))

    dtype = cp.result_type(*inputs)

    if dtype.char not in "fdgFDGO":
        raise NotImplementedError("input type '%s' not supported" % dtype)
    if zi is not None:
        zi = cp.array(zi, dtype)  # make a copy so that we can operate in place
        if zi.shape != x_zi_shape:
            raise ValueError(
                "Invalid zi shape. With axis=%r, an input with "
                "shape %r, and an sos array with %d sections, zi "
                "must have shape %r, got %r."
                % (axis, x.shape, n_sections, x_zi_shape, zi.shape)
            )
        return_zi = True
    else:
        zi = cp.zeros(x_zi_shape, dtype=dtype)
        return_zi = False

    axis = axis % x.ndim  # make positive
    x = cp.moveaxis(x, axis, -1)
    zi = cp.moveaxis(zi, [0, axis + 1], [-2, -1])
    x_shape, zi_shape = x.shape, zi.shape
    x = cp.reshape(x, (-1, x.shape[-1]))
    x = cp.array(x, dtype, order="C")  # make a copy, can modify in place
    zi = cp.ascontiguousarray(cp.reshape(zi, (-1, n_sections, 2)))
    sos = sos.astype(dtype, copy=False)

    max_smem = _get_max_smem()
    max_tpb = _get_max_tpb()

    # Determine how much shared memory is needed
    out_size = sos.shape[0]
    z_size = zi.shape[1] * zi.shape[2]
    sos_size = sos.shape[0] * sos.shape[1]
    shared_mem = (out_size + z_size + sos_size) * x.dtype.itemsize

    if shared_mem > max_smem:
        max_sections = (
            max_smem // (1 + zi.shape[2] + sos.shape[1]) // x.dtype.itemsize
        )
        raise ValueError(
            "The number of sections ({}), requires too much "
            "shared memory ({}B) > ({}B). \n"
            "\n**Max sections possible ({})**".format(
                sos.shape[0], shared_mem, max_smem, max_sections
            )
        )

    if sos.shape[0] > max_tpb:
        raise ValueError(
            "The number of sections ({}), must be less "
            "than max threads per block ({})".format(sos.shape[0], max_tpb)
        )

    if sos.shape[0] > x.shape[1]:
        raise ValueError(
            "The number of samples ({}), must be greater "
            "than the number of sections ({})".format(x.shape[1], sos.shape[0])
        )

    _sosfilt(sos, x, zi)

    x.shape = x_shape
    x = cp.moveaxis(x, -1, axis)
    if return_zi:
        zi.shape = zi_shape
        zi = cp.moveaxis(zi, [-2, -1], [0, axis + 1])
        out = (x, zi)
    else:
        out = x

    return out


_hilbert_kernel = cp.ElementwiseKernel(
    "",
    "float64 h",
    """
    if ( !odd ) {
        if ( ( i == 0 ) || ( i == bend ) ) {
            h = 1.0;
        } else if ( i > 0 && i < bend ) {
            h = 2.0;
        } else {
            h = 0.0;
        }
    } else {
        if ( i == 0 ) {
            h = 1.0;
        } else if ( i > 0 && i < bend) {
            h = 2.0;
        } else {
            h = 0.0;
        }
    }
    """,
    "_hilbert_kernel",
    options=("-std=c++11",),
    loop_prep="const bool odd { _ind.size() & 1 }; \
               const int bend = odd ? \
                   static_cast<int>( 0.5 * ( _ind.size()  + 1 ) ) : \
                   static_cast<int>( 0.5 * _ind.size() );",
)


def hilbert(x, N=None, axis=-1):
    """
    Compute the analytic signal, using the Hilbert transform.

    The transformation is done along the last axis by default.

    Parameters
    ----------
    x : array_like
        Signal data.  Must be real.
    N : int, optional
        Number of Fourier components.  Default: ``x.shape[axis]``
    axis : int, optional
        Axis along which to do the transformation.  Default: -1.

    Returns
    -------
    xa : ndarray
        Analytic signal of `x`, of each 1-D array along `axis`

    Notes
    -----
    The analytic signal ``x_a(t)`` of signal ``x(t)`` is:

    .. math:: x_a = F^{-1}(F(x) 2U) = x + i y

    where `F` is the Fourier transform, `U` the unit step function,
    and `y` the Hilbert transform of `x`. [1]_

    In other words, the negative half of the frequency spectrum is zeroed
    out, turning the real-valued signal into a complex signal.  The Hilbert
    transformed signal can be obtained from ``cp.imag(hilbert(x))``, and the
    original signal from ``cp.real(hilbert(x))``.

    Examples
    ---------
    In this example we use the Hilbert transform to determine the amplitude
    envelope and instantaneous frequency of an amplitude-modulated signal.

    >>> import cupy as cp
    >>> import matplotlib.pyplot as plt
    >>> from cusignal import hilbert, chirp

    >>> duration = 1.0
    >>> fs = 400.0
    >>> samples = int(fs*duration)
    >>> t = cp.arange(samples) / fs

    We create a chirp of which the frequency increases from 20 Hz to 100 Hz and
    apply an amplitude modulation.

    >>> signal = chirp(t, 20.0, t[-1], 100.0)
    >>> signal *= (1.0 + 0.5 * cp.sin(2.0*cp.pi*3.0*t) )

    The amplitude envelope is given by magnitude of the analytic signal. The
    instantaneous frequency can be obtained by differentiating the
    instantaneous phase in respect to time. The instantaneous phase corresponds
    to the phase angle of the analytic signal.

    >>> analytic_signal = hilbert(signal)
    >>> amplitude_envelope = cp.abs(analytic_signal)
    >>> instantaneous_phase = cp.unwrap(cp.angle(analytic_signal))
    >>> instantaneous_frequency = (cp.diff(instantaneous_phase) /
    ...                            (2.0*cp.pi) * fs)

    >>> fig = plt.figure()
    >>> ax0 = fig.add_subplot(211)
    >>> ax0.plot(cp.asnumpy(t), cp.asnumpy(signal), label='signal')
    >>> ax0.plot(cp.asnumpy(t), cp.asnumpy(amplitude_envelope), \
        label='envelope')
    >>> ax0.set_xlabel("time in seconds")
    >>> ax0.legend()
    >>> ax1 = fig.add_subplot(212)
    >>> ax1.plot(t[1:], instantaneous_frequency)
    >>> ax1.set_xlabel("time in seconds")
    >>> ax1.set_ylim(0.0, 120.0)

    References
    ----------
    .. [1] Wikipedia, "Analytic signal".
           https://en.wikipedia.org/wiki/Analytic_signal
    .. [2] Leon Cohen, "Time-Frequency Analysis", 1995. Chapter 2.
    .. [3] Alan V. Oppenheim, Ronald W. Schafer. Discrete-Time Signal
           Processing, Third Edition, 2009. Chapter 12.
           ISBN 13: 978-1292-02572-8

    """
    x = cp.asarray(x)
    if cp.iscomplexobj(x):
        raise ValueError("x must be real.")
    if N is None:
        N = x.shape[axis]
    if N <= 0:
        raise ValueError("N must be positive.")

    Xf = cp.fft.fft(x, N, axis=axis)
    h = _hilbert_kernel(size=N)

    if x.ndim > 1:
        ind = [cp.newaxis] * x.ndim
        ind[axis] = slice(None)
        h = h[tuple(ind)]
    x = cp.fft.ifft(Xf * h, axis=axis)
    return x


_hilbert2_kernel = cp.ElementwiseKernel(
    "",
    "float64 h1, float64 h2",
    """
    if ( !odd ) {
        if ( ( i == 0 ) || ( i == bend ) ) {
            h1 = 1.0;
            h2 = 1.0;
        } else if ( i > 0 && i < bend ) {
            h1 = 2.0;
            h2 = 2.0;
        } else {
            h1 = 0.0;
            h2 = 0.0;
        }
    } else {
        if ( i == 0 ) {
            h1 = 1.0;
            h2 = 1.0;
        } else if ( i > 0 && i < bend) {
            h1 = 2.0;
            h2 = 2.0;
        } else {
            h1 = 0.0;
            h2 = 0.0;
        }
    }
    """,
    "_hilbert2_kernel",
    options=("-std=c++11",),
    loop_prep="const bool odd { _ind.size() & 1 }; \
               const int bend = odd ? \
                   static_cast<int>( 0.5 * ( _ind.size()  + 1 ) ) : \
                   static_cast<int>( 0.5 * _ind.size() );",
)


def hilbert2(x, N=None):
    """
    Compute the '2-D' analytic signal of `x`

    Parameters
    ----------
    x : array_like
        2-D signal data.
    N : int or tuple of two ints, optional
        Number of Fourier components. Default is ``x.shape``

    Returns
    -------
    xa : ndarray
        Analytic signal of `x` taken along axes (0,1).

    References
    ----------
    .. [1] Wikipedia, "Analytic signal",
        https://en.wikipedia.org/wiki/Analytic_signal

    """
    x = cp.atleast_2d(x)
    if x.ndim > 2:
        raise ValueError("x must be 2-D.")
    if cp.iscomplexobj(x):
        raise ValueError("x must be real.")
    if N is None:
        N = x.shape
    elif isinstance(N, int):
        if N <= 0:
            raise ValueError("N must be positive.")
        N = (N, N)
    elif len(N) != 2 or cp.any(cp.asarray(N) <= 0):
        raise ValueError(
            "When given as a tuple, N must hold exactly two positive integers"
        )

    Xf = cp.fft.fft2(x, N, axes=(0, 1))

    h1, h2 = _hilbert2_kernel(size=N[1])

    h = h1[:, cp.newaxis] * h2[cp.newaxis, :]
    k = x.ndim
    while k > 2:
        h = h[:, cp.newaxis]
        k -= 1
    x = cp.fft.ifft2(Xf * h, axes=(0, 1))
    return x


_detrend_A_kernel = cp.ElementwiseKernel(
    "",
    "float64 A",
    """
    if ( i & 1 ) {
        const int new_i { i >> 1 };
        A = new_i * den;
    } else {
        A = 1.0;
    }
    """,
    "_detrend_A_kernel",
    options=("-std=c++11",),
    loop_prep="const double den { 1.0 / _ind.size() };",
)


def detrend(data, axis=-1, type="linear", bp=0, overwrite_data=False):
    """
    Remove linear trend along axis from data.

    Parameters
    ----------
    data : array_like
        The input data.
    axis : int, optional
        The axis along which to detrend the data. By default this is the
        last axis (-1).
    type : {'linear', 'constant'}, optional
        The type of detrending. If ``type == 'linear'`` (default),
        the result of a linear least-squares fit to `data` is subtracted
        from `data`.
        If ``type == 'constant'``, only the mean of `data` is subtracted.
    bp : array_like of ints, optional
        A sequence of break points. If given, an individual linear fit is
        performed for each part of `data` between two break points.
        Break points are specified as indices into `data`.
    overwrite_data : bool, optional
        If True, perform in place detrending and avoid a copy. Default is False

    Returns
    -------
    ret : ndarray
        The detrended input data.

    Examples
    --------
    >>> import cusignal
    >>> import cupy as cp
    >>> randgen = cp.random.RandomState(9)
    >>> npoints = 1000
    >>> noise = randgen.randn(npoints)
    >>> x = 3 + 2*cp.linspace(0, 1, npoints) + noise
    >>> (cusignal.detrend(x) - noise).max() < 0.01
    True
    """
    if type not in ["linear", "l", "constant", "c"]:
        raise ValueError("Trend type must be 'linear' or 'constant'.")
    data = cp.asarray(data)
    dtype = data.dtype.char
    if dtype not in "dfDF":
        dtype = "d"
    if type in ["constant", "c"]:
        ret = data - cp.expand_dims(cp.mean(data, axis), axis)
        return ret
    else:
        dshape = data.shape
        N = dshape[axis]
        bp = np.sort(np.unique(np.r_[0, bp, N]))
        if np.any(bp > N):
            raise ValueError(
                "Breakpoints must be less than length of \
                data along given axis."
            )

        Nreg = len(bp) - 1
        # Restructure data so that axis is along first dimension and
        #  all other dimensions are collapsed into second dimension
        rnk = len(dshape)
        if axis < 0:
            axis = axis + rnk

        newdims = np.r_[axis, 0:axis, axis + 1 : rnk]
        newdata = cp.reshape(
            cp.transpose(data, tuple(newdims)), (N, _prod(dshape) // N)
        )

        if not overwrite_data:
            newdata = newdata.copy()  # make sure we have a copy
        if newdata.dtype.char not in "dfDF":
            newdata = newdata.astype(dtype)

        # Find leastsq fit and remove it for each piece
        for m in range(Nreg):
            Npts = int(bp[m + 1] - bp[m])
            A = _detrend_A_kernel(size=Npts * 2)
            A = cp.reshape(A, (Npts, 2))
            sl = slice(bp[m], bp[m + 1])
            coef, _, _, _ = linalg.lstsq(A, newdata[sl])
            newdata[sl] = newdata[sl] - cp.dot(A, coef)

        # Put data back in original shape.
        tdshape = np.take(dshape, newdims, 0)
        ret = cp.reshape(newdata, tuple(tdshape))
        vals = list(range(1, rnk))
        olddims = vals[:axis] + [0] + vals[axis:]
        ret = cp.transpose(ret, tuple(olddims))
        return ret


_freq_shift_kernel = cp.ElementwiseKernel(
    "T x, float64 freq, float64 fs",
    "complex128 out",
    """
    thrust::complex<double> temp(0, neg2pi * freq / fs * i);
    out = x * exp(temp);
    """,
    "_freq_shift_kernel",
    options=("-std=c++11",),
    loop_prep="const double neg2pi { -1 * 2 * M_PI };",
)


def freq_shift(x, freq, fs):
    """
    Frequency shift signal by freq at fs sample rate

    Parameters
    ----------
    x : array_like, complex valued
        The data to be shifted.
    freq : float
        Shift by this many (Hz)
    fs : float
        Sampling rate of the signal
    domain : string
        freq or time
    """
    x = cp.asarray(x)
    return _freq_shift_kernel(x, freq, fs)


def channelize_poly(x, h, n_chans):
    """
    Polyphase channelize signal into n channels

    Parameters
    ----------
    x : array_like
        The input data to be channelized
    h : array_like
        The 1-D input filter; will be split into n
        channels of int number of taps
    n_chans : int
        Number of channels for channelizer

    Returns
    ----------
    yy : channelized output matrix

    Notes
    ----------
    Currently only supports simple channelizer where channel
    spacing is equivalent to the number of channels used (zero overlap).
    Number of filter taps (len of filter / n_chans) must be <=32.

    """
    dtype = cp.promote_types(x.dtype, h.dtype)

    x = cp.asarray(x, dtype=dtype)
    h = cp.asarray(h, dtype=dtype)

    # number of taps in each h_n filter
    n_taps = int(len(h) / n_chans)
    if n_taps > 32:
        raise NotImplementedError(
            "The number of calculated taps ({}) in  \
            each filter is currently capped at 32. Please reduce filter \
                length or number of channels".format(
                n_taps
            )
        )

    # number of outputs
    n_pts = int(len(x) / n_chans)

    if x.dtype == np.float32 or x.dtype == np.complex64:
        y = cp.empty((n_pts, n_chans), dtype=cp.complex64)
    elif x.dtype == np.float64 or x.dtype == np.complex128:
        y = cp.empty((n_pts, n_chans), dtype=cp.complex128)
    else:
        raise NotImplementedError(
            "Data type ({}) not allowed.".format(x.dtype)
        )

    _channelizer(x, h, y, n_chans, n_taps, n_pts)

    return cp.conj(cp.fft.fft(y)).T


def _prod(iterable):
    """
    Product of a list of numbers.
    Faster than cp.prod for short lists like array shapes.
    """
    product = 1
    for x in iterable:
        product *= x
    return product
