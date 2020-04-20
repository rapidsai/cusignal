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
import itertools
import numpy as np
import warnings

from enum import Enum
from math import ceil
from numba import complex64, complex128, cuda, float32, float64, int64, void
from string import Template

try:
    # Numba <= 0.49
    from numba.types.scalars import Complex
except ImportError:
    # Numba >= 0.49
    from numba.core.types.scalars import Complex


class GPUKernel(Enum):
    UPFIRDN = 0
    UPFIRDN2D = 1


class GPUBackend(Enum):
    CUPY = 0
    NUMBA = 1


# Numba type supported and corresponding C type
_SUPPORTED_TYPES = {
    np.float32: [float32, "float"],
    np.float64: [float64, "double"],
    np.complex64: [complex64, "complex<float>"],
    np.complex128: [complex128, "complex<double>"],
}

_numba_kernel_cache = {}
_cupy_kernel_cache = {}


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
    inp, h_trans_flip, up, down, axis, x_shape_a, h_per_phase, padded_len, out
):

    y, x = cuda.grid(2)

    if y == 0 and x == 0:
        print(x_shape_a)

    if x < out.shape[1] and y < out.shape[0]:

        if axis == 1:
            x_idx = ((x * down) // up) % padded_len
            h_idx = (x * down) % up * h_per_phase
        else:
            x_idx = ((y * down) // up) % padded_len
            h_idx = (y * down) % up * h_per_phase

        x_conv_idx = x_idx - h_per_phase + 1
        if x_conv_idx < 0:
            h_idx -= x_conv_idx
            x_conv_idx = 0

        temp: out.dtype = 0

        # If axis = 0, we need to know each column in x.
        for x_c in range(x_conv_idx, x_idx + 1):
            if x_c < x_shape_a and x_c >= 0:  # If inside input
                # if multi-dimenstional array
                if axis == 1:  # process columns
                    temp += inp[y, x_c] * h_trans_flip[h_idx]
                else:  # process rows
                    temp += inp[x_c, x] * h_trans_flip[h_idx]
                    # temp = 99

            h_idx += 1

        out[y, x] = temp


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

        temp: out.dtype = 0

        # If axis = 0, we need to know each column in x.
        for x_c in range(x_conv_idx, x_idx + 1):
            if x_c < x_shape_a and x_c >= 0:
                temp += x[x_c] * h_trans_flip[h_idx]
            h_idx += 1

        out[i] = temp


def _numba_upfirdn_signature(ty, ndim):
    if ndim == 1:
        arr_ty = ty[:]
    elif ndim == 2:
        arr_ty = ty[:, :]
    return void(
        arr_ty,  # x
        ty[:],  # h_trans_flip
        int64,  # up
        int64,  # down
        int64,  # axis
        int64,  # x_shape_a
        int64,  # h_per_phase
        int64,  # padded_len
        arr_ty,  # out
    )


# Custom Cupy raw kernel implementing upsample, filter, downsample operation
# Matthew Nicely - mnicely@nvidia.com
loaded_from_source = Template(
    """
$header

extern "C" {
    __global__ void _cupy_upfirdn_1d(
            const ${datatype} * __restrict__ inp,
            const int inpW,
            const int inpH,
            const ${datatype} * __restrict__ h_trans_flip,
            const int up,
            const int down,
            const int axis,
            const int x_shape_a,
            const int h_per_phase,
            const int padded_len,
            ${datatype} * __restrict__ out,
            const int outW,
            const int outH) {

        const int t {
            static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) };
        const int stride { static_cast<int>(blockDim.x * gridDim.x) };

        for ( int tid = t; tid < outW; tid += stride ) {
            int x_idx { static_cast<int>((tid * down) / up) % padded_len };
            int h_idx { (tid * down) % up * h_per_phase };
            int x_conv_idx { x_idx - h_per_phase + 1 };

            if ( x_conv_idx < 0 ) {
                h_idx -= x_conv_idx;
                x_conv_idx = 0;
            }

            ${datatype} temp {};

            for ( int x_c = x_conv_idx; x_c < (x_idx + 1); x_c++ ) {
                if ( x_c < x_shape_a && x_c >= 0 ) {
                    temp += inp[x_c] * h_trans_flip[h_idx];
                }
                h_idx += 1;
            }
            out[tid] = temp;
        }
    }

    __global__ void _cupy_upfirdn_2d(
            const ${datatype} * __restrict__ inp,
            const int inpW,
            const int inpH,
            const ${datatype} * __restrict__ h_trans_flip,
            const int up,
            const int down,
            const int axis,
            const int x_shape_a,
            const int h_per_phase,
            const int padded_len,
            ${datatype} * __restrict__ out,
            const int outW,
            const int outH) {


        const int ty {
            static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) };
        const int tx {
            static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) };

        if ( (tx < outH) && (ty < outW) ) {
            int x_idx {};
            int h_idx {};

            if ( axis == 1 ) {
                x_idx = ( static_cast<int>(tx * down) / up ) % padded_len;
                h_idx = (tx * down) % up * h_per_phase;
            } else {
                x_idx = ( static_cast<int>(ty * down) / up ) % padded_len;
                h_idx = (ty * down) % up * h_per_phase;
            }

            int x_conv_idx { x_idx - h_per_phase + 1 };
            if ( x_conv_idx < 0 ) {
                h_idx -= x_conv_idx;
                x_conv_idx = 0;
            }

            ${datatype} temp {};

            for ( int x_c = x_conv_idx; x_c < (x_idx + 1); x_c++ ) {
                if ( x_c < x_shape_a && x_c >= 0 ) {
                    if (axis == 1) {
                        temp += inp[ty * inpH + x_c] * h_trans_flip[h_idx];
                    } else {
                        temp += inp[x_c * inpH + tx] * h_trans_flip[h_idx];
                        //temp = 99;
                    }
                }
                h_idx += 1;
            }
            out[ty * outH + tx] = temp; // axis = 0
            //out[tx * outW + ty] = temp;
        }

    }
}
"""
)


class _cupy_upfirdn_wrapper(object):
    def __init__(self, grid, block, stream, kernel):
        if isinstance(grid, int):
            grid = (grid,)
        if isinstance(block, int):
            block = (block,)

        self.grid = grid
        self.block = block
        self.stream = stream
        self.kernel = kernel

    def __call__(
        self,
        x,
        h_trans_flip,
        up,
        down,
        axis,
        x_shape_a,
        h_per_phase,
        padded_len,
        out,
    ):

        print(x.shape, out.shape)
        if x.ndim < 2:
            x_W = x.shape[0]
            o_W = out.shape[0]
            x_H = o_H = 0
        else:
            x_W = x.shape[0]
            x_H = x.shape[1]
            o_W = out.shape[0]
            o_H = out.shape[1]

        kernel_args = (
            x,
            x_W,
            x_H,
            h_trans_flip,
            up,
            down,
            axis,
            x_shape_a,
            h_per_phase,
            padded_len,
            out,
            o_W,
            o_H,
        )

        self.stream.use()
        self.kernel(self.grid, self.block, kernel_args)


def _get_backend_kernel(dtype, grid, block, stream, use_numba, k_type):

    if not use_numba:
        kernel = _cupy_kernel_cache[(dtype.name, k_type)]
        if kernel:
            return _cupy_upfirdn_wrapper(grid, block, stream, kernel)
        else:
            raise ValueError(
                "Kernel {} not found in _cupy_kernel_cache".format(k_type)
            )

    else:
        nb_stream = stream_cupy_to_numba(stream)
        kernel = _numba_kernel_cache[(dtype.name, k_type)]
        if kernel:
            return kernel[grid, block, nb_stream]
        else:
            raise ValueError(
                "Kernel {} not found in _numba_kernel_cache".format(k_type)
            )

    raise NotImplementedError(
        "No kernel found for k_type {}, datatype {}".format(k_type, dtype.name)
    )


def _populate_kernel_cache(np_type, use_numba, k_type):

    # Check in np_type is a supported option
    try:
        numba_type, c_type = _SUPPORTED_TYPES[np_type]

    except ValueError:
        raise ValueError("No kernel found for datatype {}".format(np_type))

    # Check if use_numba is support
    try:
        GPUBackend(use_numba)

    except ValueError:
        raise

    # Check if use_numba is support
    try:
        GPUKernel(k_type)

    except ValueError:
        raise

    if not use_numba:
        if (str(numba_type), k_type) in _cupy_kernel_cache:  # Not work
            return
        # Instantiate the cupy kernel for this type and compile
        if isinstance(numba_type, Complex):
            header = "#include <cupy/complex.cuh>"
        else:
            header = ""
        src = loaded_from_source.substitute(datatype=c_type, header=header)
        module = cp.RawModule(
            code=src, options=("-std=c++11", "-use_fast_math")
        )
        if k_type == GPUKernel.UPFIRDN:
            _cupy_kernel_cache[
                (str(numba_type), k_type)
            ] = module.get_function("_cupy_upfirdn_1d")
        elif k_type == GPUKernel.UPFIRDN2D:
            _cupy_kernel_cache[
                (str(numba_type), k_type)
            ] = module.get_function("_cupy_upfirdn_2d")
        else:
            raise NotImplementedError(
                "No kernel found for k_type {}, datatype {}".format(
                    k_type, str(numba_type)
                )
            )

    else:
        if (str(numba_type), k_type) in _numba_kernel_cache:
            return
        # JIT compile the numba kernels, both 1d and 2d
        if k_type == GPUKernel.UPFIRDN:
            sig_1d = _numba_upfirdn_signature(numba_type, 1)
            _numba_kernel_cache[(str(numba_type), k_type)] = cuda.jit(
                sig_1d, fastmath=True
            )(_numba_upfirdn_1d)
        elif k_type == GPUKernel.UPFIRDN2D:
            sig_2d = _numba_upfirdn_signature(numba_type, 2)
            _numba_kernel_cache[(str(numba_type), k_type)] = cuda.jit(
                sig_2d, fastmath=True
            )(_numba_upfirdn_2d)
        else:
            raise NotImplementedError(
                "No kernel found for k_type {}, datatype {}".format(
                    k_type, str(numba_type)
                )
            )


def precompile_kernels(dtype=None, backend=None, k_type=None):
    r"""
    Precompile GPU kernels for later use.

    Parameters
    ----------
    dtype : numpy datatype or list of datatypes, optional
        Data types for which kernels should be precompiled. If not
        specified, all supported data types will be precompiled.
        Specific to this unit
            np.float32
            np.float64
            np.complex64
            np.complex128
    backend : GPUBackend, optional
        Which GPU backend to precompile for. If not specified,
        all supported backends will be precompiled.
        Specific to this unit
            GPUBackend.CUPY
            GPUBackend.NUMBA
    k_type : GPUKernel, optional
        Which GPU kernel to compile for. If not specified,
        all supported kernels will be precompiled.
        Specific to this unit
            GPUKernel.UPFIRDN
            GPUKernel.UPFIRDN2
    Examples
    ----------
    To precompile all kernels in this unit
    >>> import cusignal
    >>> from cusignal._upfirdn import GPUBackend, GPUKernel
    >>> cusignal._upfirdn.precompile_kernels()

    To precompile a specific NumPy datatype, CuPy backend, and kernel type
    >>> cusignal._upfirdn.precompile_kernels( [np.float64],
        [GPUBackend.CUPY], [GPUKernel.UPFIRDN],)


    To precompile a specific NumPy datatype and kernel type,
    but both Numba and CuPY variations
    >>> cusignal._upfirdn.precompile_kernels( dtype=[np.float64],
        k_type=[GPUKernel.UPFIRDN],)
    """

    # Ensure inputs are a list, if inputs exist
    if dtype is not None and not hasattr(dtype, "__iter__"):
        raise TypeError(
            "dtype ({}) should be in list - e.g [np.float32,]".format(dtype)
        )

    elif backend is not None and not hasattr(backend, "__iter__"):
        raise TypeError(
            "backend ({}) should be in list - e.g [{},]".format(
                backend, backend
            )
        )
    elif k_type is not None and not hasattr(k_type, "__iter__"):
        raise TypeError(
            "k_type ({}) should be in list - e.g [{},]".format(k_type, k_type)
        )
    else:
        dtype = list(dtype) if dtype else _SUPPORTED_TYPES.keys()
        backend = list(backend) if backend else list(GPUBackend)
        k_type = list(k_type) if k_type else list(GPUKernel)

        for d, b, k in itertools.product(dtype, backend, k_type):
            _populate_kernel_cache(d, b, k)


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
            blockspergrid_x = ceil(out.shape[0] / threadsperblock[0])
            blockspergrid_y = ceil(out.shape[1] / threadsperblock[1])
            blockspergrid = (blockspergrid_x, blockspergrid_y)

        else:
            device_id = cp.cuda.Device()
            numSM = device_id.attributes["MultiProcessorCount"]
            blockspergrid = numSM * 20
            threadsperblock = 512

        if out.ndim == 1:
            _populate_kernel_cache(
                out.dtype.type, use_numba, GPUKernel.UPFIRDN
            )
            kernel = _get_backend_kernel(
                out.dtype,
                blockspergrid,
                threadsperblock,
                cp_stream,
                use_numba,
                GPUKernel.UPFIRDN,
            )
        elif out.ndim == 2:
            _populate_kernel_cache(
                out.dtype.type, use_numba, GPUKernel.UPFIRDN2D
            )
            kernel = _get_backend_kernel(
                out.dtype,
                blockspergrid,
                threadsperblock,
                cp_stream,
                use_numba,
                GPUKernel.UPFIRDN2D,
            )
        else:
            raise NotImplementedError("upfirdn() requires ndim <= 2")

        kernel(
            cp.asarray(x, self._output_type),
            self._h_trans_flip,
            self._up,
            self._down,
            axis,
            x_shape_a,
            h_per_phase,
            padded_len,
            out,
        )

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
