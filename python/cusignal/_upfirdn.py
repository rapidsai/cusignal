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

import math
import warnings
from string import Template

import cupy as cp
from numba import complex64, complex128, cuda, float32, float64, int64, void
from numba.types.scalars import Complex

_numba_kernel_cache = {}
_cupy_kernel_cache = {}

# Numba type supported and corresponding C type
_SUPPORTED_TYPES = {
    float32: "float",
    float64: "double",
    complex64: "complex<float>",
    complex128: "complex<double>"
}


# Use until functionality provided in Numba 0.49/0.50 available
def stream_cupy_to_numba(cp_stream):
    """
    Notes:
        1. The lifetime of the returned Numba stream should be as
           long as the CuPy one, which handles the deallocation
           of the underlying CUDA stream.
        2. The returned Numba stream is assumed to live in the same
           CUDA context as the CuPy one.
        3. The implementation here closely follows that of
           cuda.stream() in Numba.
    """
    from ctypes import c_void_p
    import weakref

    # get the pointer to actual CUDA stream
    raw_str = cp_stream.ptr

    # gather necessary ingredients
    ctx = cuda.devices.get_context()
    handle = c_void_p(raw_str)

    # create a Numba stream
    nb_stream = cuda.cudadrv.driver.Stream(
        weakref.proxy(ctx), handle, finalizer=None
    )

    return nb_stream


def _pad_h(h, up):
    """Store coefficients in a transposed, flipped arrangement.
    For example, suppose upRate is 3, and the
    input number of coefficients is 10, represented as h[0], ..., h[9].
    Then the internal buffer will look like this::
       h[9], h[6], h[3], h[0],   // flipped phase 0 coefs
       0,    h[7], h[4], h[1],   // flipped phase 1 coefs (zero-padded)
       0,    h[8], h[5], h[2],   // flipped phase 2 coefs (zero-padded)
    """
    h_padlen = len(h) + (-len(h) % up)
    h_full = cp.zeros(h_padlen, h.dtype)
    h_full[: len(h)] = h
    h_full = h_full.reshape(-1, up).T[:, ::-1].ravel()
    return h_full


def _output_len(len_h, in_len, up, down):
    in_len_copy = in_len + (len_h + (-len_h % up)) // up - 1
    nt = in_len_copy * up
    need = nt // down
    if nt % down > 0:
        need += 1
    return need


# Custom Numba kernel implementing upsample, filter, downsample operation
# Matthew Nicely - mnicely@nvidia.com
def _numba_upfirdn_2d(
    x, h_trans_flip, up, down, axis, x_shape_a, h_per_phase, padded_len, out
):

    num_loops = 1
    for i in range(out.ndim - 1):
        if i != axis:
            num_loops *= out.shape[i]

    X, Y = cuda.grid(2)

    if X < out.shape[0] and Y < out.shape[1]:

        i = X
        y_idx = Y

        if axis == 1:
            x_idx = ((Y * down) // up) % padded_len
            h_idx = (Y * down) % up * h_per_phase
        else:
            x_idx = ((i * down) // up) % padded_len
            h_idx = (i * down) % up * h_per_phase

        x_conv_idx = x_idx - h_per_phase + 1
        if x_conv_idx < 0:
            h_idx -= x_conv_idx
            x_conv_idx = 0

        # If axis = 0, we need to know each column in x.
        for x_conv_idx in range(x_conv_idx, x_idx + 1):
            if x_conv_idx < x_shape_a and x_conv_idx >= 0:
                # if multi-dimenstional array
                if num_loops > 1:  # a loop is an additional column
                    out[i, y_idx] += x[i, x_conv_idx] * h_trans_flip[h_idx]
                else:
                    out[i, y_idx] += x[x_conv_idx, y_idx] * h_trans_flip[h_idx]

            h_idx += 1


def _numba_upfirdn_1d(
    x, h_trans_flip, up, down, axis, x_shape_a, h_per_phase, padded_len, out
):

    X = cuda.grid(1)
    strideX = cuda.gridsize(1)

    for i in range(X, cp.int32(out.shape[0]), strideX):

        x_idx = cp.int32(cp.int32(cp.int32(i * down) // up) % padded_len)
        h_idx = cp.int32(cp.int32(cp.int32(i * down) % up) * h_per_phase)

        x_conv_idx = x_idx - h_per_phase + 1
        if x_conv_idx < 0:
            h_idx -= x_conv_idx
            x_conv_idx = 0

        # If axis = 0, we need to know each column in x.
        for x_conv_idx in range(x_conv_idx, x_idx + 1):
            if x_conv_idx < x_shape_a and x_conv_idx >= 0:
                out[i] += x[x_conv_idx] * h_trans_flip[h_idx]
            h_idx += 1


def _numba_upfirdn_signature(ty, ndim):
    if ndim == 1:
        arr_ty = ty[:]
    elif ndim == 2:
        arr_ty = ty[:, :]
    return void(
        arr_ty,  # x
        ty[:],   # h_trans_flip
        int64,   # up
        int64,   # down
        int64,   # axis
        int64,   # x_shape_a
        int64,   # h_per_phase
        int64,   # padded_len
        arr_ty,  # out
    )


# Custom Cupy raw kernel implementing upsample, filter, downsample operation
# Matthew Nicely - mnicely@nvidia.com
loaded_from_source = Template(
    """
$header

extern "C" {
    __global__ void _cupy_upfirdn_1d(const int n,
            const ${datatype} * __restrict__ x,
            const ${datatype} * __restrict__ h_trans_flip,
            const int up,
            const int down,
            const int x_shape_a,
            const int h_per_phase,
            const int padded_len,
            ${datatype} * __restrict__ out) {

        const int t { blockIdx.x * blockDim.x + threadIdx.x };
        const int stride { blockDim.x * gridDim.x };

        for (int tid = t; tid < n; tid += stride) {
            int x_idx { static_cast<int>((tid * down) / up) % padded_len };
            int h_idx { (tid * down) % up * h_per_phase };
            int x_conv_idx { x_idx - h_per_phase + 1 };

            if (x_conv_idx < 0) {
                h_idx -= x_conv_idx;
                x_conv_idx = 0;
            }

            ${datatype} temp {};

            for ( int x_c = x_conv_idx; x_c < (x_idx + 1); x_c++ ) {
                if (x_c < x_shape_a && x_c >= 0) {
                    temp += x[x_c] * h_trans_flip[h_idx];
                }
                out[tid] = temp;
                h_idx += 1;
            }
        }
    }
}
"""
)


class _cupy_upfirdn_wrapper(object):

    def __init__(self, grid, block, stream, kernel):
        if isinstance(grid, int):
            grid = (grid, )
        if isinstance(block, int):
            block = (block, )

        self.grid = grid
        self.block = block
        self.stream = stream
        self.kernel = kernel

    def __call__(self,
                 x,
                 h_trans_flip,
                 up,
                 down,
                 axis,
                 x_shape_a,
                 h_per_phase,
                 padded_len,
                 out):

        kernel_args = (
            out.shape[0],
            x,
            h_trans_flip,
            up,
            down,
            x_shape_a,
            h_per_phase,
            padded_len,
            out
        )

        self.stream.use()
        self.kernel(self.grid, self.block, kernel_args)


def _get_backend_kernel(ndim, dtype, grid, block, stream, use_numba):
    if ndim > 2:
        raise NotImplementedError(
            "upfirdn() requires ndim <= 2")
    elif ndim > 1 and not use_numba:
        warnings.warn(
            "CuPy backend is only implemented for ndim == 1 \
                Running with Numba CUDA backend", UserWarning)
        use_numba = True

    if use_numba:
        nb_stream = stream_cupy_to_numba(stream)
        kernel = _numba_kernel_cache[(ndim, dtype.name)]
        if kernel:
            return kernel[grid, block, nb_stream]
    else:
        kernel = _cupy_kernel_cache[(ndim, dtype.name)]
        if kernel:
            return _cupy_upfirdn_wrapper(grid, block, stream, kernel)

    raise NotImplementedError(
        "No kernel found for ndim {}, datatype {}".format(ndim, dtype.name))


def _populate_kernel_cache():
    for numba_type, c_type in _SUPPORTED_TYPES.items():
        # JIT compile the numba kernels, both 1d and 2d
        sig_1d = _numba_upfirdn_signature(numba_type, 1)
        _numba_kernel_cache[(1, str(numba_type))] = \
            cuda.jit(sig_1d, fastmath=True)(_numba_upfirdn_1d)

        sig_2d = _numba_upfirdn_signature(numba_type, 2)
        _numba_kernel_cache[(2, str(numba_type))] = \
            cuda.jit(sig_2d, fastmath=True)(_numba_upfirdn_2d)

        # Instantiate the cupy kernel for this type and compile
        if isinstance(numba_type, Complex):
            header = "#include <cupy/complex.cuh>"
        else:
            header = ""
        src = loaded_from_source.substitute(datatype=c_type, header=header)
        module2 = cp.RawModule(code=src,
                               options=("-std=c++11", "-use_fast_math"))
        _cupy_kernel_cache[(1, str(numba_type))] = \
            module2.get_function("_cupy_upfirdn_1d")


class _UpFIRDn(object):
    def __init__(self, h, x_dtype, up, down):
        """Helper for resampling"""
        h = cp.asarray(h)
        if h.ndim != 1 or h.size == 0:
            raise ValueError("h must be 1D with non-zero length")

        self._output_type = cp.result_type(h.dtype, x_dtype, cp.float32)
        h = cp.asarray(h, self._output_type)
        self._up = int(up)
        self._down = int(down)
        if self._up < 1 or self._down < 1:
            raise ValueError("Both up and down must be >= 1")
        # This both transposes, and "flips" each phase for filtering
        self._h_trans_flip = _pad_h(h, self._up)
        self._h_trans_flip = cp.ascontiguousarray(self._h_trans_flip)

    def apply_filter(
        self,
        x,
        axis=-1,
        cp_stream=cp.cuda.stream.Stream(null=True),
        use_numba=False,
    ):
        """Apply the prepared filter to the specified axis of a nD signal x"""

        output_len = _output_len(
            len(self._h_trans_flip), x.shape[axis], self._up, self._down
        )
        output_shape = cp.asarray(x.shape)
        output_shape[axis] = output_len
        out = cp.zeros(
            cp.asnumpy(output_shape), dtype=self._output_type, order="C"
        )
        axis = axis % x.ndim

        # Precompute variables on CPU
        x_shape_a = x.shape[axis]
        h_per_phase = len(self._h_trans_flip) // self._up
        padded_len = x.shape[axis] + (len(self._h_trans_flip) // self._up) - 1

        if out.ndim > 1:
            threadsperblock = (16, 16)
            blockspergrid_x = math.ceil(out.shape[0] / threadsperblock[0])
            blockspergrid_y = math.ceil(out.shape[1] / threadsperblock[1])
            blockspergrid = (blockspergrid_x, blockspergrid_y)

        else:
            device_id = cp.cuda.Device()
            numSM = device_id.attributes["MultiProcessorCount"]
            blockspergrid = numSM * 20
            threadsperblock = 512

        kernel = _get_backend_kernel(out.ndim,
                                     out.dtype,
                                     blockspergrid,
                                     threadsperblock,
                                     cp_stream,
                                     use_numba)

        kernel(cp.asarray(x, self._output_type),
               self._h_trans_flip,
               self._up,
               self._down,
               axis,
               x_shape_a,
               h_per_phase,
               padded_len,
               out)

        return out


def upfirdn(
    h,
    x,
    up=1,
    down=1,
    axis=-1,
    cp_stream=cp.cuda.stream.Stream(null=True),
    use_numba=False,
):
    """Upsample, FIR filter, and downsample
    Parameters
    ----------
    h : array_like
        1-dimensional FIR (finite-impulse response) filter coefficients.
    x : array_like
        Input signal array.
    up : int, optional
        Upsampling rate. Default is 1.
    down : int, optional
        Downsampling rate. Default is 1.
    axis : int, optional
        The axis of the input data array along which to apply the
        linear filter. The filter is applied to each subarray along
        this axis. Default is -1.
    cp_stream : CuPy stream, optional
        Option allows upfirdn to run in a non-default stream. The use
        of multiple non-default streams allow multiple kernels to
        run concurrently. Default is cp.cuda.stream.Stream(null=True)
        or default stream.
    use_numba : bool, optional
        Option to use Numba CUDA kernel or raw CuPy kernel. Raw CuPy
        can yield performance gains over Numba. Default is False.
    Returns
    -------
    y : ndarray
        The output signal array. Dimensions will be the same as `x` except
        for along `axis`, which will change size according to the `h`,
        `up`,  and `down` parameters.
    Notes
    -----
    The algorithm is an implementation of the block diagram shown on page 129
    of the Vaidyanathan text [1]_ (Figure 4.3-8d).
    .. [1] P. P. Vaidyanathan, Multirate Systems and Filter Banks,
       Prentice Hall, 1993.
    The direct approach of upsampling by factor of P with zero insertion,
    FIR filtering of length ``N``, and downsampling by factor of Q is
    O(N*Q) per output sample. The polyphase implementation used here is
    O(N/P).
    .. versionadded:: 0.18
    Examples
    --------
    Simple operations:
    >>> from scipy.signal import upfirdn
    >>> upfirdn([1, 1, 1], [1, 1, 1])   # FIR filter
    array([ 1.,  2.,  3.,  2.,  1.])
    >>> upfirdn([1], [1, 2, 3], 3)  # upsampling with zeros insertion
    array([ 1.,  0.,  0.,  2.,  0.,  0.,  3.,  0.,  0.])
    >>> upfirdn([1, 1, 1], [1, 2, 3], 3)  # upsampling with sample-and-hold
    array([ 1.,  1.,  1.,  2.,  2.,  2.,  3.,  3.,  3.])
    >>> upfirdn([.5, 1, .5], [1, 1, 1], 2)  # linear interpolation
    array([ 0.5,  1. ,  1. ,  1. ,  1. ,  1. ,  0.5,  0. ])
    >>> upfirdn([1], np.arange(10), 1, 3)  # decimation by 3
    array([ 0.,  3.,  6.,  9.])
    >>> upfirdn([.5, 1, .5], np.arange(10), 2, 3)  # linear interp, rate 2/3
    array([ 0. ,  1. ,  2.5,  4. ,  5.5,  7. ,  8.5,  0. ])
    Apply a single filter to multiple signals:
    >>> x = np.reshape(np.arange(8), (4, 2))
    >>> x
    array([[0, 1],
           [2, 3],
           [4, 5],
           [6, 7]])
    Apply along the last dimension of ``x``:
    >>> h = [1, 1]
    >>> upfirdn(h, x, 2)
    array([[ 0.,  0.,  1.,  1.],
           [ 2.,  2.,  3.,  3.],
           [ 4.,  4.,  5.,  5.],
           [ 6.,  6.,  7.,  7.]])
    Apply along the 0th dimension of ``x``:
    >>> upfirdn(h, x, 2, axis=0)
    array([[ 0.,  1.],
           [ 0.,  1.],
           [ 2.,  3.],
           [ 2.,  3.],
           [ 4.,  5.],
           [ 4.,  5.],
           [ 6.,  7.],
           [ 6.,  7.]])
    """
    x = cp.asarray(x)
    ufd = _UpFIRDn(h, x.dtype, up, down)
    # This is equivalent to (but faster than) using np.apply_along_axis
    return ufd.apply_filter(x, axis, cp_stream=cp_stream, use_numba=use_numba)


# Code executed at import time goes here to make it clear to future
# maintainers the side-effects of importing this module.

# 1) Load and compile upfirdn kernels for each supported data type.
_populate_kernel_cache()
