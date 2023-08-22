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

import cupy as cp

_gauss_spline_kernel = cp.ElementwiseKernel(
    "T x, int32 n",
    "T output",
    """
    output = 1 / sqrt( 2.0 * M_PI * signsq ) * exp( -( x * x ) * r_signsq );
    """,
    "_gauss_spline_kernel",
    options=("-std=c++11",),
    loop_prep="const double signsq { ( n + 1 ) / 12.0 }; \
               const double r_signsq { 0.5 / signsq };",
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

    return _gauss_spline_kernel(x, n)


_cubic_kernel = cp.ElementwiseKernel(
    "T x",
    "T res",
    """
    const T ax { abs( x ) };

    if( ax < 1 ) {
        res =  2.0 / 3 - 1.0 / 2  * ax * ax * ( 2.0 - ax );
    } else if( !( ax < 1 ) && ( ax < 2 ) ) {
        res = 1.0 / 6 * ( 2.0 - ax ) *  ( 2.0 - ax ) * ( 2.0 - ax );
    } else {
        res = 0.0;
    }
    """,
    "_cubic_kernel",
    options=("-std=c++11",),
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
    const T ax { abs( x ) };

    if( ax < 0.5 ) {
        res = 0.75 - ax * ax;
    } else if( !( ax < 0.5 ) && ( ax < 1.5 ) ) {
        res = ( ( ax - 1.5 ) * ( ax - 1.5 ) ) * 0.5 ;
    } else {
        res = 0.0;
    }
    """,
    "_quadratic_kernel",
    options=("-std=c++11",),
)


def quadratic(x):
    """A quadratic B-spline.

    This is a special case of `bspline`, and equivalent to ``bspline(x, 2)``.
    """
    x = cp.asarray(x)

    return _quadratic_kernel(x)
