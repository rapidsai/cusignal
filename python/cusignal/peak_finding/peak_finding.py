# Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# cuSignal does not support cupy.take(mode='foo')

import cupy as cp

from ._peak_finding_cuda import _peak_finding


def _boolrelextrema(data, comparator, axis=0, order=1, mode="clip"):
    """
    Calculate the relative extrema of `data`.

    Relative extrema are calculated by finding locations where
    ``comparator(data[n], data[n+1:n+order+1])`` is True.

    Parameters
    ----------
    data : ndarray
        Array in which to find the relative extrema.
    comparator : callable
        Function to use to compare two data points.
        Should take two arrays as arguments.
    axis : int, optional
        Axis over which to select from `data`.  Default is 0.
    order : int, optional
        How many points on each side to use for the comparison
        to consider ``comparator(n,n+x)`` to be True.

    Returns
    -------
    extrema : ndarray
        Boolean array of the same shape as `data` that is True at an extrema,
        False otherwise.

    See also
    --------
    argrelmax, argrelmin
    """
    if (int(order) != order) or (order < 1):
        raise ValueError("Order must be an int >= 1")

    if data.ndim < 3:
        results = cp.empty(data.shape, dtype=bool)
        _peak_finding(data, comparator, axis, order, mode, results)
    else:
        datalen = data.shape[axis]
        locs = cp.arange(0, datalen)
        results = cp.ones(data.shape, dtype=bool)
        main = cp.take(data, locs, axis=axis)
        for shift in cp.arange(1, order + 1):
            if mode == "clip":
                p_locs = cp.clip(locs + shift, a_min=None, a_max=(datalen - 1))
                m_locs = cp.clip(locs - shift, a_min=0, a_max=None)
            else:
                p_locs = locs + shift
                m_locs = locs - shift
            plus = cp.take(data, p_locs, axis=axis)
            minus = cp.take(data, m_locs, axis=axis)
            results &= comparator(main, plus)
            results &= comparator(main, minus)

            if ~results.any():
                return results

    return results


def argrelmin(data, axis=0, order=1, mode="clip"):
    """
    Calculate the relative minima of `data`.

    Parameters
    ----------
    data : ndarray
        Array in which to find the relative minima.
    axis : int, optional
        Axis over which to select from `data`.  Default is 0.
    order : int, optional
        How many points on each side to use for the comparison
        to consider ``comparator(n, n+x)`` to be True.

    Returns
    -------
    extrema : tuple of ndarrays
        Indices of the minima in arrays of integers.  ``extrema[k]`` is
        the array of indices of axis `k` of `data`.  Note that the
        return value is a tuple even when `data` is one-dimensional.

    See Also
    --------
    argrelextrema, argrelmax, find_peaks

    Notes
    -----
    This function uses `argrelextrema` with cp.less as comparator. Therefore it
    requires a strict inequality on both sides of a value to consider it a
    minimum. This means flat minima (more than one sample wide) are not
    detected. In case of one-dimensional `data` `find_peaks` can be used to
    detect all local minima, including flat ones, by calling it with negated
    `data`.

    Examples
    --------
    >>> from cusignal import argrelmin
    >>> import cupy as cp
    >>> x = cp.array([2, 1, 2, 3, 2, 0, 1, 0])
    >>> argrelmin(x)
    (array([1, 5]),)
    >>> y = cp.array([[1, 2, 1, 2],
    ...               [2, 2, 0, 0],
    ...               [5, 3, 4, 4]])
    ...
    >>> argrelmin(y, axis=1)
    (array([0, 2]), array([2, 1]))

    """
    data = cp.asarray(data)
    return argrelextrema(data, cp.less, axis, order, mode)


def argrelmax(data, axis=0, order=1, mode="clip"):
    """
    Calculate the relative maxima of `data`.

    Parameters
    ----------
    data : ndarray
        Array in which to find the relative maxima.
    axis : int, optional
        Axis over which to select from `data`.  Default is 0.
    order : int, optional
        How many points on each side to use for the comparison
        to consider ``comparator(n, n+x)`` to be True.

    Returns
    -------
    extrema : tuple of ndarrays
        Indices of the maxima in arrays of integers.  ``extrema[k]`` is
        the array of indices of axis `k` of `data`.  Note that the
        return value is a tuple even when `data` is one-dimensional.

    See Also
    --------
    argrelextrema, argrelmin, find_peaks

    Notes
    -----
    This function uses `argrelextrema` with cp.greater as comparator. Therefore
    it  requires a strict inequality on both sides of a value to consider it a
    maximum. This means flat maxima (more than one sample wide) are not
    detected. In case of one-dimensional `data` `find_peaks` can be used to
    detect all local maxima, including flat ones.

    Examples
    --------
    >>> from cusignal import argrelmax
    >>> import cupy as cp
    >>> x = cp.array([2, 1, 2, 3, 2, 0, 1, 0])
    >>> argrelmax(x)
    (array([3, 6]),)
    >>> y = cp.array([[1, 2, 1, 2],
    ...               [2, 2, 0, 0],
    ...               [5, 3, 4, 4]])
    ...
    >>> argrelmax(y, axis=1)
    (array([0]), array([1]))
    """
    data = cp.asarray(data)
    return argrelextrema(data, cp.greater, axis, order, mode)


def argrelextrema(data, comparator, axis=0, order=1, mode="clip"):
    """
    Calculate the relative extrema of `data`.

    Parameters
    ----------
    data : ndarray
        Array in which to find the relative extrema.
    comparator : callable
        Function to use to compare two data points.
        Should take two arrays as arguments.
    axis : int, optional
        Axis over which to select from `data`.  Default is 0.
    order : int, optional
        How many points on each side to use for the comparison
        to consider ``comparator(n, n+x)`` to be True.

    Returns
    -------
    extrema : tuple of ndarrays
        Indices of the maxima in arrays of integers.  ``extrema[k]`` is
        the array of indices of axis `k` of `data`.  Note that the
        return value is a tuple even when `data` is one-dimensional.

    See Also
    --------
    argrelmin, argrelmax

    Examples
    --------
    >>> from cusignal import argrelextrema
    >>> import cupy as cp
    >>> x = cp.array([2, 1, 2, 3, 2, 0, 1, 0])
    >>> argrelextrema(x, cp.greater)
    (array([3, 6]),)
    >>> y = cp.array([[1, 2, 1, 2],
    ...               [2, 2, 0, 0],
    ...               [5, 3, 4, 4]])
    ...
    >>> argrelextrema(y, cp.less, axis=1)
    (array([0, 2]), array([2, 1]))

    """
    data = cp.asarray(data)
    results = _boolrelextrema(data, comparator, axis, order, mode)

    if mode == "raise":
        raise NotImplementedError("CuPy `take` doesn't support `mode='raise'`.")

    return cp.nonzero(results)
