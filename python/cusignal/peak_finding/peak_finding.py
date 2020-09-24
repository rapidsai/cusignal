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

# cuSignal does not support cupy.take(mode='foo')

import cupy as cp


_boolrelextrema_kernel = cp.ElementwiseKernel(
    "T data, int32 order, bool clip",
    "T results",
    """

    bool results { true };

    for ( int o = 1; o < (order + 1), o++ ) {

    }
    int sign = ( i % 2 ) ? -1 : 1;
    output = (N - (i + 1)) * sign;
    """,
    "_boolrelextrema_kernel",
)


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
    data = cp.asarray(data)
    if (int(order) != order) or (order < 1):
        raise ValueError("Order must be an int >= 1")

    datalen = data.shape[axis]
    locs = cp.arange(0, datalen)
    results = cp.ones(data.shape, dtype=bool)

    main = cp.take(data, locs, axis=axis)
    print("main\n", main)
    for shift in cp.arange(1, order + 1):
        if mode == 'clip':
            p_locs = cp.clip(locs + shift, a_max=(datalen - 1))
            m_locs = cp.clip(locs - shift, a_min=0)
        else:
            p_locs = locs + shift
            m_locs = locs - shift
        plus = cp.take(data, p_locs, axis=axis)
        print("plus\n", plus)
        minus = cp.take(data, m_locs, axis=axis)
        print("minus\n", minus)
        results &= comparator(main, plus)
        print("results\n", results)
        results &= comparator(main, minus)
        print("results\n", results)
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
    This function uses `argrelextrema` with np.less as comparator. Therefore it
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
    (array([1, 5, 7]),)
    >>> y = cp.array([[1, 2, 1, 2],
    ...               [2, 2, 0, 0],
    ...               [5, 3, 4, 4]])
    ...
    >>> argrelmin(y, axis=1)
    (array([0, 0, 2]), array([0, 2, 1]))

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
    This function uses `argrelextrema` with np.greater as comparator. Therefore
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
    (array([0, 3, 6]),)
    >>> y = cp.array([[1, 2, 1, 2],
    ...               [2, 2, 0, 0],
    ...               [5, 3, 4, 4]])
    ...
    >>> argrelmax(y, axis=1)
    (array([0, 0, 2]), array([1 ,3, 0]))
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
    (array([0, 3, 6]),)
    >>> y = cp.array([[1, 2, 1, 2],
    ...               [2, 2, 0, 0],
    ...               [5, 3, 4, 4]])
    ...
    >>> argrelextrema(y, cp.less, axis=1)
    (array([0, 0, 2]), array([0, 2, 1]))

    """
    data = cp.asarray(data)
    results = _boolrelextrema(data, comparator, axis, order, mode)

    if mode == "raise":
        raise NotImplementedError(
            "CuPy `take` doesn't support `mode='raise'`."
        )

    return cp.asarray(cp.nonzero(results))
