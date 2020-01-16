# cuSignal does not support cupy.take(mode='foo')

import cupy as cp


def _boolrelextrema(data, comparator, axis=0, order=1):
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

    Examples
    --------
    >>> testdata = np.array([1,2,3,2,1])
    >>> _boolrelextrema(testdata, np.greater, axis=0)
    array([False, False,  True, False, False], dtype=bool)

    """
    data = cp.asarray(data)
    if((int(order) != order) or (order < 1)):
        raise ValueError('Order must be an int >= 1')

    datalen = data.shape[axis]
    locs = cp.arange(0, datalen)

    results = cp.ones(data.shape, dtype=bool)
    main = data.take(locs, axis=axis)
    for shift in cp.arange(1, order + 1):
        plus = data.take(locs + shift, axis=axis)
        minus = data.take(locs - shift, axis=axis)
        results &= comparator(main, plus)
        results &= comparator(main, minus)
        if(~results.any()):
            return results
    return results


def argrelmin(data, axis=0, order=1):
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

    .. versionadded:: 0.11.0

    Examples
    --------
    >>> from scipy.signal import argrelmin
    >>> x = np.array([2, 1, 2, 3, 2, 0, 1, 0])
    >>> argrelmin(x)
    (array([1, 5]),)
    >>> y = np.array([[1, 2, 1, 2],
    ...               [2, 2, 0, 0],
    ...               [5, 3, 4, 4]])
    ...
    >>> argrelmin(y, axis=1)
    (array([0, 2]), array([2, 1]))

    """
    data = cp.asarray(data)
    return argrelextrema(data, cp.less, axis, order)


def argrelmax(data, axis=0, order=1):
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

    .. versionadded:: 0.11.0

    Examples
    --------
    >>> from scipy.signal import argrelmax
    >>> x = np.array([2, 1, 2, 3, 2, 0, 1, 0])
    >>> argrelmax(x)
    (array([3, 6]),)
    >>> y = np.array([[1, 2, 1, 2],
    ...               [2, 2, 0, 0],
    ...               [5, 3, 4, 4]])
    ...
    >>> argrelmax(y, axis=1)
    (array([0]), array([1]))
    """
    data = cp.asarray(data)
    return argrelextrema(data, cp.greater, axis, order)


def argrelextrema(data, comparator, axis=0, order=1):
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

    Notes
    -----

    .. versionadded:: 0.11.0

    Examples
    --------
    >>> from scipy.signal import argrelextrema
    >>> x = np.array([2, 1, 2, 3, 2, 0, 1, 0])
    >>> argrelextrema(x, np.greater)
    (array([3, 6]),)
    >>> y = np.array([[1, 2, 1, 2],
    ...               [2, 2, 0, 0],
    ...               [5, 3, 4, 4]])
    ...
    >>> argrelextrema(y, np.less, axis=1)
    (array([0, 2]), array([2, 1]))

    """
    data = cp.asarray(data)
    results = _boolrelextrema(data, comparator,
                              axis, order)
    return cp.nonzero(results)
