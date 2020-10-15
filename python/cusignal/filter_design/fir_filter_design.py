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

from scipy import signal
from ..windows.windows import get_window


def _get_fs(fs, nyq):
    """
    Utility for replacing the argument 'nyq' (with default 1) with 'fs'.
    """
    if nyq is None and fs is None:
        fs = 2
    elif nyq is not None:
        if fs is not None:
            raise ValueError("Values cannot be given for both 'nyq' and 'fs'.")
        fs = 2 * nyq
    return fs


# Some notes on function parameters:
#
# `cutoff` and `width` are given as numbers between 0 and 1.  These are
# relative frequencies, expressed as a fraction of the Nyquist frequency.
# For example, if the Nyquist frequency is 2 KHz, then width=0.15 is a width
# of 300 Hz.
#
# The `order` of a FIR filter is one less than the number of taps.
# This is a potential source of confusion, so in the following code,
# we will always use the number of taps as the parameterization of
# the 'size' of the filter. The "number of taps" means the number
# of coefficients, which is the same as the length of the impulse
# response of the filter.


def kaiser_beta(a):
    """Compute the Kaiser parameter `beta`, given the attenuation `a`.
    Parameters
    ----------
    a : float
        The desired attenuation in the stopband and maximum ripple in
        the passband, in dB.  This should be a *positive* number.
    Returns
    -------
    beta : float
        The `beta` parameter to be used in the formula for a Kaiser window.
    References
    ----------
    Oppenheim, Schafer, "Discrete-Time Signal Processing", p.475-476.
    """
    if a > 50:
        beta = 0.1102 * (a - 8.7)
    elif a > 21:
        beta = 0.5842 * (a - 21) ** 0.4 + 0.07886 * (a - 21)
    else:
        beta = 0.0
    return beta


def kaiser_atten(numtaps, width):
    """Compute the attenuation of a Kaiser FIR filter.
    Given the number of taps `N` and the transition width `width`, compute the
    attenuation `a` in dB, given by Kaiser's formula:
        a = 2.285 * (N - 1) * pi * width + 7.95
    Parameters
    ----------
    numtaps : int
        The number of taps in the FIR filter.
    width : float
        The desired width of the transition region between passband and
        stopband (or, in general, at any discontinuity) for the filter.
    Returns
    -------
    a : float
        The attenuation of the ripple, in dB.
    """
    a = 2.285 * (numtaps - 1) * np.pi * width + 7.95
    return a


_firwin_kernel = cp.ElementwiseKernel(
    "float64 win, int32 numtaps, raw float64 bands, int32 steps, bool scale",
    "float64 h, float64 hc",
    """
    double m { static_cast<double>( i ) - alpha ?
        static_cast<double>( i ) - alpha : 1.0e-20 };

    double temp {};
    double left {};
    double right {};

    for ( int s = 0; s < steps; s++ ) {
        left = bands[s * 2 + 0] ? bands[s * 2 + 0] : 1.0e-20;
        right = bands[s * 2 + 1] ? bands[s * 2 + 1] : 1.0e-20;

        temp += right * ( sin( right * m * M_PI ) / ( right * m * M_PI ) );
        temp -= left * ( sin( left * m * M_PI ) / ( left * m * M_PI ) );
    }

    temp *= win;
    h = temp;

    double scale_frequency {};

    if ( scale ) {
        left = bands[0];
        right = bands[1];

        if ( left == 0 ) {
            scale_frequency = 0.0;
        } else if ( right == 1 ) {
            scale_frequency = 1.0;
        } else {
            scale_frequency = 0.5 * ( left + right );
        }
        double c { cos( M_PI * m * scale_frequency ) };
        hc = temp * c;
    }
    """,
    "_firwin_kernel",
    options=("-std=c++11",),
    loop_prep="const double alpha { 0.5 * ( numtaps - 1 ) };",
)


def firwin(
    numtaps,
    cutoff,
    width=None,
    window="hamming",
    pass_zero=True,
    scale=True,
    nyq=1.0,
    fs=None,
    gpupath=True,
):
    """
    FIR filter design using the window method.

    This function computes the coefficients of a finite impulse response
    filter.  The filter will have linear phase; it will be Type I if
    `numtaps` is odd and Type II if `numtaps` is even.

    Type II filters always have zero response at the Nyquist frequency, so a
    ValueError exception is raised if firwin is called with `numtaps` even and
    having a passband whose right end is at the Nyquist frequency.

    Parameters
    ----------
    numtaps : int
        Length of the filter (number of coefficients, i.e. the filter
        order + 1).  `numtaps` must be odd if a passband includes the
        Nyquist frequency.
    cutoff : float or 1D array_like
        Cutoff frequency of filter (expressed in the same units as `fs`)
        OR an array of cutoff frequencies (that is, band edges). In the
        latter case, the frequencies in `cutoff` should be positive and
        monotonically increasing between 0 and `fs/2`.  The values 0 and
        `fs/2` must not be included in `cutoff`.
    width : float or None, optional
        If `width` is not None, then assume it is the approximate width
        of the transition region (expressed in the same units as `fs`)
        for use in Kaiser FIR filter design.  In this case, the `window`
        argument is ignored.
    window : string or tuple of string and parameter values, optional
        Desired window to use. See `cusignal.get_window` for a list
        of windows and required parameters.
    pass_zero : {True, False, 'bandpass', 'lowpass', 'highpass', 'bandstop'},
        optional
        If True, the gain at the frequency 0 (i.e. the "DC gain") is 1.
        If False, the DC gain is 0. Can also be a string argument for the
        desired filter type (equivalent to ``btype`` in IIR design functions).

        .. versionadded:: 1.3.0
           Support for string arguments.
    scale : bool, optional
        Set to True to scale the coefficients so that the frequency
        response is exactly unity at a certain frequency.
        That frequency is either:

        - 0 (DC) if the first passband starts at 0 (i.e. pass_zero
          is True)
        - `fs/2` (the Nyquist frequency) if the first passband ends at
          `fs/2` (i.e the filter is a single band highpass filter);
          center of first passband otherwise

    nyq : float, optional
        *Deprecated.  Use `fs` instead.*  This is the Nyquist frequency.
        Each frequency in `cutoff` must be between 0 and `nyq`. Default
        is 1.
    fs : float, optional
        The sampling frequency of the signal.  Each frequency in `cutoff`
        must be between 0 and ``fs/2``.  Default is 2.
    gpupath : bool, Optional
        Optional path for filter design. gpupath == False may be desirable if
        filter sizes are small.

    Returns
    -------
    h : (numtaps,) ndarray
        Coefficients of length `numtaps` FIR filter.

    Raises
    ------
    ValueError
        If any value in `cutoff` is less than or equal to 0 or greater
        than or equal to ``fs/2``, if the values in `cutoff` are not strictly
        monotonically increasing, or if `numtaps` is even but a passband
        includes the Nyquist frequency.

    See Also
    --------
    firwin2
    firls
    minimum_phase
    remez

    Examples
    --------
    Low-pass from 0 to f:

    >>> import cusignal
    >>> numtaps = 3
    >>> f = 0.1
    >>> cusignal.firwin(numtaps, f)
    array([ 0.06799017,  0.86401967,  0.06799017])

    Use a specific window function:

    >>> cusignal.firwin(numtaps, f, window='nuttall')
    array([  3.56607041e-04,   9.99286786e-01,   3.56607041e-04])

    High-pass ('stop' from 0 to f):

    >>> cusignal.firwin(numtaps, f, pass_zero=False)
    array([-0.00859313,  0.98281375, -0.00859313])

    Band-pass:

    >>> f1, f2 = 0.1, 0.2
    >>> cusignal.firwin(numtaps, [f1, f2], pass_zero=False)
    array([ 0.06301614,  0.88770441,  0.06301614])

    Band-stop:

    >>> cusignal.firwin(numtaps, [f1, f2])
    array([-0.00801395,  1.0160279 , -0.00801395])

    Multi-band (passbands are [0, f1], [f2, f3] and [f4, 1]):

    >>> f3, f4 = 0.3, 0.4
    >>> cusignal.firwin(numtaps, [f1, f2, f3, f4])
    array([-0.01376344,  1.02752689, -0.01376344])

    Multi-band (passbands are [f1, f2] and [f3,f4]):

    >>> cusignal.firwin(numtaps, [f1, f2, f3, f4], pass_zero=False)
    array([ 0.04890915,  0.91284326,  0.04890915])

    """
    if gpupath:
        pp = cp
    else:
        pp = np

    cutoff = pp.atleast_1d(cutoff) / float(nyq)

    # print("cutoff", cutoff.size)

    # Check for invalid input.
    if cutoff.ndim > 1:
        raise ValueError(
            "The cutoff argument must be at most " "one-dimensional."
        )
    if cutoff.size == 0:
        raise ValueError("At least one cutoff frequency must be given.")
    if cutoff.min() <= 0 or cutoff.max() >= 1:
        raise ValueError(
            "Invalid cutoff frequency: frequencies must be "
            "greater than 0 and less than nyq."
        )
    if pp.any(pp.diff(cutoff) <= 0):
        raise ValueError(
            "Invalid cutoff frequencies: the frequencies "
            "must be strictly increasing."
        )

    if width is not None:
        # A width was given.  Find the beta parameter of the Kaiser window
        # and set `window`.  This overrides the value of `window` passed in.
        atten = kaiser_atten(numtaps, float(width) / nyq)
        beta = kaiser_beta(atten)
        window = ("kaiser", beta)

    pass_nyquist = bool(cutoff.size & 1) ^ pass_zero
    if pass_nyquist and numtaps % 2 == 0:
        raise ValueError(
            "A filter with an even number of coefficients must "
            "have zero response at the Nyquist rate."
        )

    # Insert 0 and/or 1 at the ends of cutoff so that the length of cutoff
    # is even, and each pair in cutoff corresponds to passband.
    cutoff = pp.hstack(([0.0] * pass_zero, cutoff, [1.0] * pass_nyquist))

    # `bands` is a 2D array; each row gives the left and right edges of
    # a passband.
    bands = cutoff.reshape(-1, 2)

    if gpupath:
        win = get_window(window, numtaps, fftbins=False)
        h, hc = _firwin_kernel(win, numtaps, bands, bands.shape[0], scale)
        if scale:
            s = cp.sum(hc)
            h /= s
    else:
        try:
            win = signal.get_window(window, numtaps, fftbins=False)
        except NameError:
            raise RuntimeError("CPU path requires SciPy Signal's get_windows.")

        # Build up the coefficients.
        alpha = 0.5 * (numtaps - 1)
        m = np.arange(0, numtaps) - alpha
        h = 0
        for left, right in bands:
            h += right * np.sinc(right * m)
            h -= left * np.sinc(left * m)

        h *= win

        # Now handle scaling if desired.
        if scale:
            # Get the first passband.
            left, right = bands[0]
            if left == 0:
                scale_frequency = 0.0
            elif right == 1:
                scale_frequency = 1.0
            else:
                scale_frequency = 0.5 * (left + right)
            c = np.cos(np.pi * m * scale_frequency)
            s = np.sum(h * c)
            h /= s

    return h


def cmplx_sort(p):
    """Sort roots based on magnitude.

    Parameters
    ----------
    p : array_like
        The roots to sort, as a 1-D array.

    Returns
    -------
    p_sorted : ndarray
        Sorted roots.
    indx : ndarray
        Array of indices needed to sort the input `p`.

    Examples
    --------
    >>> import cusignal
    >>> vals = [1, 4, 1+1.j, 3]
    >>> p_sorted, indx = cusignal.cmplx_sort(vals)
    >>> p_sorted
    array([1.+0.j, 1.+1.j, 3.+0.j, 4.+0.j])
    >>> indx
    array([0, 2, 3, 1])

    """
    p = cp.asarray(p)
    if cp.iscomplexobj(p):
        indx = cp.argsort(abs(p))
    else:
        indx = cp.argsort(p)
    return cp.take(p, indx, 0), indx
