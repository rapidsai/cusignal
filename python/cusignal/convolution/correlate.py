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

from .convolve import convolve
from .convolution_utils import _reverse_and_conj, _inputs_swap_needed
from .. import _signaltools

_modedict = {"valid": 0, "same": 1, "full": 2}


def correlate(
    in1,
    in2,
    mode="full",
    method="auto",
    cp_stream=cp.cuda.stream.Stream(null=True),
    autosync=True,
):
    r"""
    Cross-correlate two N-dimensional arrays.

    Cross-correlate `in1` and `in2`, with the output size determined by the
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
           The output is the full discrete linear cross-correlation
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
           must be at least as large as the other in every dimension.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.
    method : str {'auto', 'direct', 'fft'}, optional
        A string indicating which method to use to calculate the correlation.

        ``direct``
           The correlation is determined directly from sums, the definition of
           correlation.
        ``fft``
           The Fast Fourier Transform is used to perform the correlation more
           quickly (only available for numerical arrays.)
        ``auto``
           Automatically chooses direct or Fourier method based on an estimate
           of which is faster (default).  See `convolve` Notes for more detail.
    cp_stream : CuPy stream, optional
        Option allows upfirdn to run in a non-default stream. The use
        of multiple non-default streams allow multiple kernels to
        run concurrently. Default is cp.cuda.stream.Stream(null=True)
        or default stream.
    autosync : bool, optional
        Option to automatically synchronize cp_stream. This will block
        the host code until kernel is finished on the GPU. Setting to
        false will allow asynchronous operation but might required
        manual synchronize later `cp_stream.synchronize()`
        Default is True.

    Returns
    -------
    correlate : array
        An N-dimensional array containing a subset of the discrete linear
        cross-correlation of `in1` with `in2`.

    See Also
    --------
    choose_conv_method : contains more documentation on `method`.

    Notes
    -----
    The correlation z of two d-dimensional arrays x and y is defined as::

        z[...,k,...] =
            sum[..., i_l, ...] x[..., i_l,...] * conj(y[..., i_l - k,...])

    This way, if x and y are 1-D arrays and ``z = correlate(x, y, 'full')``
    then

    .. math::

          z[k] = (x * y)(k - N + 1)
               = \sum_{l=0}^{||x||-1}x_l y_{l-k+N-1}^{*}

    for :math:`k = 0, 1, ..., ||x|| + ||y|| - 2`

    where :math:`||x||` is the length of ``x``, :math:`N = \max(||x||,||y||)`,
    and :math:`y_m` is 0 when m is outside the range of y.

    ``method='fft'`` only works for numerical arrays as it relies on
    `fftconvolve`. In certain cases (i.e., arrays of objects or when
    rounding integers can lose precision), ``method='direct'`` is always used.

    Examples
    --------
    Implement a matched filter using cross-correlation, to recover a signal
    that has passed through a noisy channel.

    >>> import cusignal
    >>> import cupy as cp
    >>> sig = cp.repeat([0., 1., 1., 0., 1., 0., 0., 1.], 128)
    >>> sig_noise = sig + cp.random.randn(len(sig))
    >>> corr = cusignal.correlate(sig_noise, cp.ones(128), mode='same') / 128

    >>> import matplotlib.pyplot as plt
    >>> clock = cp.arange(64, len(sig), 128)
    >>> fig, (cp.asnumpy(ax_orig), cp.asnumpy(ax_noise), \
        cp.asnumpy(ax_corr)) = plt.subplots(3, 1, sharex=True)
    >>> ax_orig.plot(sig)
    >>> ax_orig.plot(clock, sig[clock], 'ro')
    >>> ax_orig.set_title('Original signal')
    >>> ax_noise.plot(cp.asnumpy(sig_noise))
    >>> ax_noise.set_title('Signal with noise')
    >>> ax_corr.plot(cp.asnumpy(corr))
    >>> ax_corr.plot(cp.asnumpy(clock), cp.asnumpy(corr[clock]), 'ro')
    >>> ax_corr.axhline(0.5, ls=':')
    >>> ax_corr.set_title('Cross-correlated with rectangular pulse')
    >>> ax_orig.margins(0, 0.1)
    >>> fig.tight_layout()
    >>> fig.show()

    """
    in1 = cp.asarray(in1)
    in2 = cp.asarray(in2)

    if in1.ndim == in2.ndim == 0:
        return in1 * in2.conj()
    elif in1.ndim != in2.ndim:
        raise ValueError("in1 and in2 should have the same dimensionality")

    # this either calls fftconvolve or this function with method=='direct'
    if method in ("fft", "auto"):
        return convolve(in1, _reverse_and_conj(in2), mode, method)

    elif method == "direct":

        if in1.ndim > 1:
            raise ValueError("Direct method is only implemented for 1D")

        swapped_inputs = in2.size > in1.size

        if swapped_inputs:
            in1, in2 = in2, in1

        return _signaltools._convolve(
            in1, in2, False, swapped_inputs, mode, cp_stream, autosync
        )

    else:
        raise ValueError(
            "Acceptable method flags are 'auto'," " 'direct', or 'fft'."
        )


def correlate2d(
    in1,
    in2,
    mode="full",
    boundary="fill",
    fillvalue=0,
    cp_stream=cp.cuda.stream.Stream(null=True),
    autosync=True,
    use_numba=False,
):
    """
    Cross-correlate two 2-dimensional arrays.
    Cross correlate `in1` and `in2` with output size determined by `mode`, and
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
           The output is the full discrete linear cross-correlation
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
    cp_stream : CuPy stream, optional
        Option allows upfirdn to run in a non-default stream. The use
        of multiple non-default streams allow multiple kernels to
        run concurrently. Default is cp.cuda.stream.Stream(null=True)
        or default stream.
    autosync : bool, optional
        Option to automatically synchronize cp_stream. This will block
        the host code until kernel is finished on the GPU. Setting to
        false will allow asynchronous operation but might required
        manual synchronize later `cp_stream.synchronize()`
        Default is true.
    use_numba : bool, optional
        Option to use Numba CUDA kernel or raw CuPy kernel. Raw CuPy
        can yield performance gains over Numba. Default is False.

    Returns
    -------
    correlate2d : ndarray
        A 2-dimensional array containing a subset of the discrete linear
        cross-correlation of `in1` with `in2`.
    Examples
    --------
    Use 2D cross-correlation to find the location of a template in a noisy
    image:
    >>> import cusignal
    >>> import cupy as cp
    >>> from scipy import misc
    >>> face = cp.asarray(misc.face(gray=True) - misc.face(gray=True).mean())
    >>> template = cp.copy(face[300:365, 670:750])  # right eye
    >>> template -= template.mean()
    >>> face = face + cp.random.randn(*face.shape) * 50  # add noise
    >>> corr = cusignal.correlate2d(face, template, boundary='symm', \
        mode='same')
    >>> y, x = cp.unravel_index(cp.argmax(corr), corr.shape)  # find the match
    >>> import matplotlib.pyplot as plt
    >>> fig, (cp.asnumpy(ax_orig), cp.asnumpy(ax_template), \
        cp.asnumpy(ax_corr)) = plt.subplots(3, 1, figsize=(6, 15))
    >>> ax_orig.imshow(cp.asnumpy(face), cmap='gray')
    >>> ax_orig.set_title('Original')
    >>> ax_orig.set_axis_off()
    >>> ax_template.imshow(cp.asnumpy(template), cmap='gray')
    >>> ax_template.set_title('Template')
    >>> ax_template.set_axis_off()
    >>> ax_corr.imshow(cp.asnumpy(corr), cmap='gray')
    >>> ax_corr.set_title('Cross-correlation')
    >>> ax_corr.set_axis_off()
    >>> ax_orig.plot(cp.asnumpy(x), cp.asnumpy(y), 'ro')
    >>> fig.show()
    """
    in1 = cp.asarray(in1)
    in2 = cp.asarray(in2)

    if not in1.ndim == in2.ndim == 2:
        raise ValueError("correlate2d inputs must both be 2D arrays")

    swapped_inputs = _inputs_swap_needed(mode, in1.shape, in2.shape)
    if swapped_inputs:
        in1, in2 = in2, in1

    out = _signaltools._convolve2d(
        in1,
        in2.conj(),
        0,
        mode,
        boundary,
        fillvalue,
        cp_stream,
        autosync,
        use_numba,
    )

    if swapped_inputs:
        out = out[::-1, ::-1]

    return out
