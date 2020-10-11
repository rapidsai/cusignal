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
    output = 1 / sqrt( 2 * pi * signsq ) * exp( -( x * x ) * r_siqnsq );
    """,
    "_gauss_spline_kernel",
    options=('-std=c++11',)
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
    T ax { abs( x ) };

    if( ax < 1 ) {
        res =  2.0 / 3 - 1.0 / 2  * ax * ax * ( 2 - ax );
    } else if( !( ax < 1 ) && ( ax < 2 ) ) {
        res = 1.0 / 6 * ( 2 - ax ) *  ( 2 - ax ) * ( 2 - ax );
    } else {
        res = 0;
    }
    """,
    "_cubic_kernel",
    options=('-std=c++11',)
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
    T ax { abs( x ) };

    if( ax < 0.5 ) {
        res = 0.75 - ax * ax;
    } else if( !( ax < 0.5 ) && ( ax < 1.5 ) ) {
        res = ( ( ax - 1.5 ) * ( ax - 1.5 ) ) * 0.5 ;
    } else {
        res = 0;
    }
    """,
    "_quadratic_kernel",
    options=('-std=c++11',)
)


def quadratic(x):
    """A quadratic B-spline.

    This is a special case of `bspline`, and equivalent to ``bspline(x, 2)``.
    """
    x = cp.asarray(x)

    return _quadratic_kernel(x)
