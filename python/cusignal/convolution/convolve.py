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

import sys

import cupy as cp
import numpy as np

from ..utils.fftpack_helper import _init_nd_shape_and_axes_sorted, next_fast_len
from . import _convolution_cuda
from .convolution_utils import (
    _centered,
    _fftconv_faster,
    _inputs_swap_needed,
    _numeric_arrays,
    _timeit_fast,
)

_modedict = {"valid": 0, "same": 1, "full": 2}


def convolve(
    in1,
    in2,
    mode="full",
    method="auto",
):
    """
    Convolve two N-dimensional arrays.

    Convolve `in1` and `in2`, with the output size determined by the
    `mode` argument.

    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:

        ``full``
           The output is the full discrete linear convolution
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
           must be at least as large as the other in every dimension.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.
    method : str {'auto', 'direct', 'fft'}, optional
        A string indicating which method to use to calculate the convolution.

        ``direct``
           The convolution is determined directly from sums, the definition of
           convolution.
        ``fft``
           The Fourier Transform is used to perform the convolution by calling
           `fftconvolve`.
        ``auto``
           Automatically chooses direct or Fourier method based on an estimate
           of which is faster (default).

    Returns
    -------
    convolve : array
        An N-dimensional array containing a subset of the discrete linear
        convolution of `in1` with `in2`.

    See Also
    --------
    choose_conv_method : chooses the fastest appropriate convolution method
    fftconvolve

    Notes
    -----
    By default, `convolve` and `correlate` use ``method='auto'``, which calls
    `choose_conv_method` to choose the fastest method using pre-computed
    values (`choose_conv_method` can also measure real-world timing with a
    keyword argument). Because `fftconvolve` relies on floating point numbers,
    there are certain constraints that may force `method=direct` (more detail
    in `choose_conv_method` docstring).

    Examples
    --------
    Smooth a square pulse using a Hann window:

    >>> import cusignal
    >>> import cupy as cp
    >>> sig = cp.repeat(cp.asarray([0., 1., 0.]), 100)
    >>> win = cusignal.hann(50)
    >>> filtered = cusignal.convolve(sig, win, mode='same') / cp.sum(win)

    >>> import matplotlib.pyplot as plt
    >>> fig, (ax_orig, ax_win, ax_filt) = plt.subplots(3, 1, sharex=True)
    >>> ax_orig.plot(cp.asnumpy(sig))
    >>> ax_orig.set_title('Original pulse')
    >>> ax_orig.margins(0, 0.1)
    >>> ax_win.plot(cp.asnumpy(win))
    >>> ax_win.set_title('Filter impulse response')
    >>> ax_win.margins(0, 0.1)
    >>> ax_filt.plot(cp.asnumpy(filtered))
    >>> ax_filt.set_title('Filtered signal')
    >>> ax_filt.margins(0, 0.1)
    >>> fig.tight_layout()
    >>> fig.show()

    """

    volume = cp.asarray(in1)
    kernel = cp.asarray(in2)

    if volume.ndim == kernel.ndim == 0:
        return volume * kernel
    elif volume.ndim != kernel.ndim:
        raise ValueError("in1 and in2 should have the same dimensionality")

    if _inputs_swap_needed(mode, volume.shape, kernel.shape):
        # Convolution is commutative
        # order doesn't have any effect on output
        volume, kernel = kernel, volume

    if method == "auto":
        method = choose_conv_method(volume, kernel, mode=mode)

    if method == "fft":
        out = fftconvolve(volume, kernel, mode=mode)
        result_type = cp.result_type(volume, kernel)
        if result_type.kind in {"u", "i"}:
            out = cp.around(out)
        return out.astype(result_type)
    elif method == "direct":
        if volume.ndim > 1:
            raise ValueError("Direct method is only implemented for 1D")

        swapped_inputs = (mode != "valid") and (kernel.size > volume.size)

        if swapped_inputs:
            volume, kernel = kernel, volume

        return _convolution_cuda._convolve(volume, kernel, True, swapped_inputs, mode)

    else:
        raise ValueError("Acceptable method flags are 'auto'," " 'direct', or 'fft'.")


def fftconvolve(in1, in2, mode="full", axes=None):
    """Convolve two N-dimensional arrays using FFT.

    Convolve `in1` and `in2` using the fast Fourier transform method, with
    the output size determined by the `mode` argument.

    This is generally much faster than `convolve` for large arrays (n > ~500),
    but can be slower when only a few output values are needed, and can only
    output float arrays (int or object array inputs will be cast to float).

    As of v0.19, `convolve` automatically chooses this method or the direct
    method based on an estimation of which is faster.

    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:

        ``full``
           The output is the full discrete linear convolution
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
           must be at least as large as the other in every dimension.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.
           axis : tuple, optional
    axes : int or array_like of ints or None, optional
        Axes over which to compute the convolution.
        The default is over all axes.

    Returns
    -------
    out : array
        An N-dimensional array containing a subset of the discrete linear
        convolution of `in1` with `in2`.

    Examples
    --------
    Autocorrelation of white noise is an impulse.

    >>> import cusignal
    >>> import cupy as cp
    >>> import numpy as np
    >>> sig = cp.random.randn(1000)
    >>> autocorr = cusignal.fftconvolve(sig, sig[::-1], mode='full')

    >>> import matplotlib.pyplot as plt
    >>> fig, (ax_orig, ax_mag) = plt.subplots(2, 1)
    >>> ax_orig.plot(cp.asnumpy(sig))
    >>> ax_orig.set_title('White noise')
    >>> ax_mag.plot(cp.asnumpy(cp.arange(-len(sig)+1,len(sig))),
            cp.asnumpy(autocorr))
    >>> ax_mag.set_title('Autocorrelation')
    >>> fig.tight_layout()
    >>> fig.show()

    Gaussian blur implemented using FFT convolution.  Notice the dark borders
    around the image, due to the zero-padding beyond its boundaries.
    The `convolve2d` function allows for other types of image boundaries,
    but is far slower.

    >>> from scipy import misc
    >>> face = misc.face(gray=True)
    >>> kernel = cp.outer(cusignal.gaussian(70, 8), cusignal.gaussian(70, 8))
    >>> blurred = cusignal.convolve(face, kernel, mode='same')

    >>> fig, (ax_orig, ax_kernel, ax_blurred) = plt.subplots(3, 1,
    ...                                                      figsize=(6, 15))
    >>> ax_orig.imshow(face, cmap='gray')
    >>> ax_orig.set_title('Original')
    >>> ax_orig.set_axis_off()
    >>> ax_kernel.imshow(cp.asnumpy(kernel), cmap='gray')
    >>> ax_kernel.set_title('Gaussian kernel')
    >>> ax_kernel.set_axis_off()
    >>> ax_blurred.imshow(cp.asnumpy(blurred), cmap='gray')
    >>> ax_blurred.set_title('Blurred')
    >>> ax_blurred.set_axis_off()
    >>> fig.show()

    """
    in1 = cp.ascontiguousarray(in1)
    in2 = cp.ascontiguousarray(in2)
    noaxes = axes is None

    if in1.ndim == in2.ndim == 0:  # scalar inputs
        return in1 * in2
    elif in1.ndim != in2.ndim:
        raise ValueError("in1 and in2 should have the same dimensionality")
    elif in1.size == 0 or in2.size == 0:  # empty arrays
        return cp.array([])

    _, axes = _init_nd_shape_and_axes_sorted(in1, shape=None, axes=axes)
    # axes needs to be numpy type for proper execution in FFT
    axes = cp.asnumpy(axes)

    if not noaxes and not axes.size:
        raise ValueError("when provided, axes cannot be empty")

    if noaxes:
        other_axes = np.array([], dtype=cp.intc)
    else:
        other_axes = np.setdiff1d(np.arange(in1.ndim), axes)

    s1 = np.array(in1.shape)
    s2 = np.array(in2.shape)

    if not np.all(
        (s1[other_axes] == s2[other_axes])
        | (s1[other_axes] == 1)
        | (s2[other_axes] == 1)
    ):
        raise ValueError(
            "incompatible shapes for in1 and in2:"
            " {0} and {1}".format(in1.shape, in2.shape)
        )

    complex_result = np.issubdtype(in1.dtype, np.complexfloating) or np.issubdtype(
        in2.dtype, cp.complexfloating
    )
    shape = np.maximum(s1, s2)
    shape[axes] = s1[axes] + s2[axes] - 1

    # Check that input sizes are compatible with 'valid' mode
    if _inputs_swap_needed(mode, s1, s2):
        # Convolution is commutative; order doesn't have any effect on output
        in1, s1, in2, s2 = in2, s2, in1, s1

    # Speed up FFT by padding to optimal size for FFTPACK
    fshape = [next_fast_len(d) for d in shape[axes]]
    fslice = tuple([slice(sz) for sz in shape])

    if not complex_result:
        sp1 = cp.fft.rfftn(in1, fshape, axes=axes)
        sp2 = cp.fft.rfftn(in2, fshape, axes=axes)
        ret = cp.fft.irfftn(sp1 * sp2, fshape, axes=axes)[fslice].copy()
    else:
        sp1 = cp.fft.fftn(in1, fshape, axes=axes)
        sp2 = cp.fft.fftn(in2, fshape, axes=axes)
        ret = cp.fft.ifftn(sp1 * sp2, axes=axes)[fslice].copy()

    if mode == "full":
        return ret
    elif mode == "same":
        return _centered(ret, s1)
    elif mode == "valid":
        shape_valid = shape.copy()
        shape_valid[axes] = s1[axes] - s2[axes] + 1
        return _centered(ret, shape_valid)
    else:
        raise ValueError(
            "acceptable mode flags are \
                        'valid',"
            " 'same', or 'full'"
        )


def convolve2d(
    in1,
    in2,
    mode="full",
    boundary="fill",
    fillvalue=0,
):
    """
    Convolve two 2-dimensional arrays.
    Convolve `in1` and `in2` with output size determined by `mode`, and
    boundary conditions determined by `boundary` and `fillvalue`.
    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:
        ``full``
           The output is the full discrete linear convolution
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
           must be at least as large as the other in every dimension.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.
    boundary : str {'fill', 'wrap', 'symm'}, optional
        A flag indicating how to handle boundaries:
        ``fill``
           pad input arrays with fillvalue. (default)
        ``wrap``
           circular boundary conditions.
        ``symm``
           symmetrical boundary conditions.
    fillvalue : scalar, optional
        Value to fill pad input arrays with. Default is 0.

    Returns
    -------
    out : ndarray
        A 2-dimensional array containing a subset of the discrete linear
        convolution of `in1` with `in2`.

    Examples
    --------
    Compute the gradient of an image by 2D convolution with a complex Scharr
    operator.  (Horizontal operator is real, vertical is imaginary.)  Use
    symmetric boundary condition to avoid creating edges at the image
    boundaries.

    >>> import cusignal
    >>> import cupy as cp
    >>> from scipy import misc
    >>> ascent = cp.asarray(misc.ascent())
    >>> scharr = cp.array([[ -3-3j, 0-10j,  +3 -3j],
    ...                    [-10+0j, 0+ 0j, +10 +0j],
    ...                    [ -3+3j, 0+10j,  +3 +3j]]) # Gx + j*Gy
    >>> grad = cusignal.convolve2d(ascent, scharr, boundary='symm', \
                mode='same')
    >>> import matplotlib.pyplot as plt
    >>> fig, (ax_orig, ax_mag, ax_ang) = plt.subplots(3, 1, figsize=(6, 15))
    >>> ax_orig.imshow(cp.asnumpy(ascent), cmap='gray')
    >>> ax_orig.set_title('Original')
    >>> ax_orig.set_axis_off()
    >>> ax_mag.imshow(cp.asnumpy(cp.absolute(grad)), cmap='gray')
    >>> ax_mag.set_title('Gradient magnitude')
    >>> ax_mag.set_axis_off()
    >>> ax_ang.imshow(cp.asnumpy(cp.angle(grad)), cmap='hsv')
    >>> ax_ang.set_title('Gradient orientation')
    >>> ax_ang.set_axis_off()
    >>> fig.show()

    """

    in1 = cp.asarray(in1)
    in2 = cp.asarray(in2)

    if not in1.ndim == in2.ndim == 2:
        raise ValueError("convolve2d inputs must both be 2D arrays")

    if _inputs_swap_needed(mode, in1.shape, in2.shape):
        in1, in2 = in2, in1

    return _convolution_cuda._convolve2d(
        in1,
        in2,
        1,
        mode,
        boundary,
        fillvalue,
    )


def choose_conv_method(in1, in2, mode="full", measure=False):
    """
    Find the fastest convolution/correlation method.

    This primarily exists to be called during the ``method='auto'`` option in
    `convolve` and `correlate`, but can also be used when performing many
    convolutions of the same input shapes and dtypes, determining
    which method to use for all of them, either to avoid the overhead of the
    'auto' option or to use accurate real-world measurements.

    Parameters
    ----------
    in1 : array_like
        The first argument passed into the convolution function.
    in2 : array_like
        The second argument passed into the convolution function.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:

        ``full``
           The output is the full discrete linear convolution
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.
    measure : bool, optional
        If True, run and time the convolution of `in1` and `in2` with both
        methods and return the fastest. If False (default), predict the fastest
        method using precomputed values.

    Returns
    -------
    method : str
        A string indicating which convolution method is fastest, either
        'direct' or 'fft'
    times : dict, optional
        A dictionary containing the times (in seconds) needed for each method.
        This value is only returned if ``measure=True``.

    See Also
    --------
    convolve
    correlate

    Examples
    --------
    Estimate the fastest method for a given input:

    >>> import cusignal
    >>> import cupy as cp
    >>> a = cp.random.randn(1000)
    >>> b = cp.random.randn(1000000)
    >>> method = cusignal.choose_conv_method(a, b, mode='same')
    >>> method
    'fft'

    This can then be applied to other arrays of the same dtype and shape:

    >>> c = cp.random.randn(1000)
    >>> d = cp.random.randn(1000000)
    >>> # `method` works with correlate and convolve
    >>> corr1 = cusignal.correlate(a, b, mode='same', method=method)
    >>> corr2 = cusignal.correlate(c, d, mode='same', method=method)
    >>> conv1 = cusignal.convolve(a, b, mode='same', method=method)
    >>> conv2 = cusignal.convolve(c, d, mode='same', method=method)

    """
    volume = cp.asarray(in1)
    kernel = cp.asarray(in2)

    if measure:
        times = {}
        for method in ("fft", "direct"):
            times[method] = _timeit_fast(
                lambda: convolve(volume, kernel, mode=mode, method=method)
            )

        chosen_method = "fft" if times["fft"] < times["direct"] else "direct"
        return chosen_method, times

    # fftconvolve doesn't support complex256
    fftconv_unsup = "complex256" if sys.maxsize > 2**32 else "complex192"
    if hasattr(cp, fftconv_unsup):
        if volume.dtype == fftconv_unsup or kernel.dtype == fftconv_unsup:
            return "direct"

    # for integer input,
    # catch when more precision required than float provides (representing an
    # integer as float can lose precision in fftconvolve if larger than 2**52)
    if any([_numeric_arrays([x], kinds="ui") for x in [volume, kernel]]):
        max_value = int(cp.abs(volume).max()) * int(cp.abs(kernel).max())
        max_value *= int(min(volume.size, kernel.size))
        if max_value > 2 ** cp.finfo("float").nmant - 1:
            return "direct"

    if _numeric_arrays([volume, kernel], kinds="b"):
        return "direct"

    if _numeric_arrays([volume, kernel]):
        if _fftconv_faster(volume, kernel, mode):
            return "fft"

    return "direct"


def convolve1d2o(
    in1,
    in2,
    mode="valid",
    method="direct",
):
    """
    Convolve a 1-dimensional arrays with a 2nd order filter.
    This results in a second order convolution.

    Convolve `in1` and `in2`, with the output size determined by the
    `mode` argument.

    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:

        ``full``
           The output is the full discrete linear convolution
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
           must be at least as large as the other in every dimension.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.
    method : str {'auto', 'direct', 'fft'}, optional
        A string indicating which method to use to calculate the convolution.

        ``direct``
           The convolution is determined directly from sums, the definition of
           convolution.
        ``fft``
           The Fourier Transform is used to perform the convolution by calling
           `fftconvolve`.
        ``auto``
           Automatically chooses direct or Fourier method based on an estimate
           of which is faster (default).

    Returns
    -------
    out : ndarray
        A 1-dimensional array containing a subset of the discrete linear
        convolution of `in1` with `in2`.

    See Also
    --------
    convolve
    convolve1d2o
    convolve1d3o

    Examples
    --------
    Convolution of a 2nd order filter on a 1d signal

    >>> import cusignal as cs
    >>> import numpy as np
    >>> d = 50
    >>> a = np.random.uniform(-1,1,(200))
    >>> b = np.random.uniform(-1,1,(d,d))
    >>> c = cs.convolve1d2o(a,b)

    """

    signal = cp.asarray(in1)
    kernel = cp.asarray(in2)

    if mode == "valid" and signal.shape[0] < kernel.shape[0]:
        # Convolution is commutative
        # order doesn't have any effect on output
        signal, kernel = kernel, signal

    if mode in ["same", "full"]:
        raise NotImplementedError("Mode == {} not implemented".format(mode))

    if method == "direct":
        return _convolution_cuda._convolve1d2o(signal, kernel, mode)
    else:
        raise NotImplementedError("Only Direct method implemented")


def convolve1d3o(
    in1,
    in2,
    mode="valid",
    method="direct",
):
    """
    Convolve a 1-dimensional array with a 3rd order filter.
    This results in a second order convolution.

    Convolve `in1` and `in2`, with the output size determined by the
    `mode` argument.

    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:

        ``full``
           The output is the full discrete linear convolution
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
           must be at least as large as the other in every dimension.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.
    method : str {'auto', 'direct', 'fft'}, optional
        A string indicating which method to use to calculate the convolution.

        ``direct``
           The convolution is determined directly from sums, the definition of
           convolution.
        ``fft``
           The Fourier Transform is used to perform the convolution by calling
           `fftconvolve`.
        ``auto``
           Automatically chooses direct or Fourier method based on an estimate
           of which is faster (default).

    Returns
    -------
    out : ndarray
        A 1-dimensional array containing a subset of the discrete linear
        convolution of `in1` with `in2`.

    See Also
    --------
    convolve
    convolve1d2o
    convolve1d3o

    Examples
    --------
    Convolution of a 3rd order filter on a 1d signal

    >>> import cusignal as cs
    >>> import numpy as np
    >>> d = 50
    >>> a = np.random.uniform(-1,1,(200))
    >>> b = np.random.uniform(-1,1,(d,d,d))
    >>> c = cs.convolve1d3o(a,b)

    """

    signal = cp.asarray(in1)
    kernel = cp.asarray(in2)

    if mode == "valid" and signal.shape[0] < kernel.shape[0]:
        # Convolution is commutative
        # order doesn't have any effect on output
        signal, kernel = kernel, signal

    if mode in ["same", "full"]:
        raise NotImplementedError("Mode == {} not implemented".format(mode))

    if method == "direct":
        return _convolution_cuda._convolve1d3o(signal, kernel, mode)
    else:
        raise NotImplementedError("Only Direct method implemented")
