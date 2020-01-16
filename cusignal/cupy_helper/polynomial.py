# Implemention of functionals not currently supported in cuPy v6
# poly1d type unsupported

from cupy import arange, asarray, zeros_like


def polyval(p, x):
    """
    Evaluate a polynomial at specific values.
    If `p` is of length N, this function returns the value:
        ``p[0]*x**(N-1) + p[1]*x**(N-2) + ... + p[N-2]*x + p[N-1]``
    If `x` is a sequence, then `p(x)` is returned for each element of `x`.
    If `x` is another polynomial then the composite polynomial `p(x(t))`
    is returned.
    Parameters
    ----------
    p : array_like or poly1d object
       1D array of polynomial coefficients (including coefficients equal
       to zero) from highest degree to the constant term, or an
       instance of poly1d.
    x : array_like or poly1d object
       A number, an array of numbers, or an instance of poly1d, at
       which to evaluate `p`.
    Returns
    -------
    values : ndarray or poly1d
       If `x` is a poly1d instance, the result is the composition of the two
       polynomials, i.e., `x` is "substituted" in `p` and the simplified
       result is returned. In addition, the type of `x` - array_like or
       poly1d - governs the type of the output: `x` array_like => `values`
       array_like, `x` a poly1d object => `values` is also.
    See Also
    --------
    poly1d: A polynomial class.
    Notes
    -----
    Horner's scheme [1]_ is used to evaluate the polynomial. Even so,
    for polynomials of high degree the values may be inaccurate due to
    rounding errors. Use carefully.
    References
    ----------
    .. [1] I. N. Bronshtein, K. A. Semendyayev, and K. A. Hirsch (Eng.
       trans. Ed.), *Handbook of Mathematics*, New York, Van Nostrand
       Reinhold Co., 1985, pg. 720.
    Examples
    --------
    >>> np.polyval([3,0,1], 5)  # 3 * 5**2 + 0 * 5**1 + 1
    76
    >>> np.polyval([3,0,1], np.poly1d(5))
    poly1d([ 76.])
    >>> np.polyval(np.poly1d([3,0,1]), 5)
    76
    >>> np.polyval(np.poly1d([3,0,1]), np.poly1d(5))
    poly1d([ 76.])
    """
    p = asarray(p)
    x = asarray(x)
    y = zeros_like(x)
    for i in arange(0, len(p)):
        y = y * x + p[i]
    return y
