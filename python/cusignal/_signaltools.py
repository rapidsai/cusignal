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

from enum import Enum
from math import gcd
from numba import (
    complex64,
    complex128,
    cuda,
    float32,
    float64,
    int32,
    int64,
    void,
)
from string import Template

try:
    # Numba <= 0.49
    from numba.types.scalars import Complex
except ImportError:
    # Numba >= 0.49
    from numba.core.types.scalars import Complex

from .fir_filter_design import firwin


class GPUKernel(Enum):
    CORRELATE = 0
    CONVOLVE = 1
    CORRELATE2D = 2
    CONVOLVE2D = 3


class GPUBackend(Enum):
    CUPY = 0
    NUMBA = 1


# Numba type supported and corresponding C type
_SUPPORTED_TYPES = {
    np.int32: [int32, "int"],
    np.int64: [int64, "long int"],
    np.float32: [float32, "float"],
    np.float64: [float64, "double"],
    np.complex64: [complex64, "complex<float>"],
    np.complex128: [complex128, "complex<double>"],
}

_numba_kernel_cache = {}
_cupy_kernel_cache = {}

FULL = 2
SAME = 1
VALID = 0

CIRCULAR = 8
REFLECT = 4
PAD = 0

_modedict = {"valid": 0, "same": 1, "full": 2}

_boundarydict = {
    "fill": 0,
    "pad": 0,
    "wrap": 2,
    "circular": 2,
    "symm": 1,
    "symmetric": 1,
    "reflect": 4,
}


def _valfrommode(mode):
    try:
        return _modedict[mode]
    except KeyError:
        raise ValueError(
            "Acceptable mode flags are 'valid'," " 'same', or 'full'."
        )


def _bvalfromboundary(boundary):
    try:
        return _boundarydict[boundary] << 2
    except KeyError:
        raise ValueError(
            "Acceptable boundary flags are 'fill', 'circular' "
            "(or 'wrap'), and 'symmetric' (or 'symm')."
        )


def _iDivUp(a, b):
    return (a // b + 1) if (a % b != 0) else (a // b)


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


def _numba_correlate_2d(
    inp, inpW, inpH, kernel, S0, S1, out, outW, outH, pick
):

    y, x = cuda.grid(2)

    if pick != 3:  # non-square
        i = cp.int32(x + S0)
    else:
        i = cp.int32(x + S1)
    j = cp.int32(y + S0)

    oPixelPos = (x, y)
    if (x < outH) and (y < outW):

        temp: out.dtype = 0

        if pick == 1:  # odd
            for k in range(cp.int32(-S0), cp.int32(S0 + 1)):
                for l in range(cp.int32(-S0), cp.int32(S0 + 1)):
                    iPixelPos = (cp.int32(i + k), cp.int32(j + l))
                    coefPos = (cp.int32(k + S0), cp.int32(l + S0))
                    temp += inp[iPixelPos] * kernel[coefPos]
                    # temp = x

        elif pick == 2:  # even
            for k in range(cp.int32(-S0), cp.int32(S0)):
                for l in range(cp.int32(-S0), cp.int32(S0)):
                    iPixelPos = (cp.int32(i + k), cp.int32(j + l))
                    coefPos = (cp.int32(k + S0), cp.int32(l + S0))
                    temp += inp[iPixelPos] * kernel[coefPos]

        else:  # non-squares
            for k in range(cp.int32(S0)):
                for l in range(cp.int32(S1)):
                    iPixelPos = (
                        cp.int32(cp.int32(i + k) - S1),
                        cp.int32(cp.int32(j + l) - S0),
                    )
                    coefPos = (k, l)
                    temp += inp[iPixelPos] * kernel[coefPos]

        out[oPixelPos] = temp


def _numba_convolve_2d(inp, inpW, inpH, kernel, S0, S1, out, outW, outH, pick):

    y, x = cuda.grid(2)

    if pick != 3:  # non-square
        i = cp.int32(x + S0)
    else:
        i = cp.int32(x + S1)
    j = cp.int32(y + S0)

    oPixelPos = (x, y)
    if (x < outH) and (y < outW):

        temp: out.dtype = 0

        if pick == 1:  # odd
            for k in range(cp.int32(-S0), cp.int32(S0 + 1)):
                for l in range(cp.int32(-S0), cp.int32(S0 + 1)):
                    iPixelPos = (cp.int32(i + k), cp.int32(j + l))
                    coefPos = (cp.int32(-k + S0), cp.int32(-l + S0))
                    temp += inp[iPixelPos] * kernel[coefPos]

        elif pick == 2:  # even
            for k in range(cp.int32(-S0), cp.int32(S0)):
                for l in range(cp.int32(-S0), cp.int32(S0)):
                    iPixelPos = (cp.int32(i + k), cp.int32(j + l))
                    coefPos = (
                        cp.int32(cp.int32(-k + S0) - 1),
                        cp.int32(cp.int32(-l + S0) - 1),
                    )
                    temp += inp[iPixelPos] * kernel[coefPos]

        else:  # non-squares
            for k in range(cp.int32(S0)):
                for l in range(cp.int32(S1)):
                    iPixelPos = (
                        cp.int32(cp.int32(i + k) - S1),
                        cp.int32(cp.int32(j + l) - S0),
                    )
                    coefPos = (
                        cp.int32(cp.int32(-k + S0) - 1),
                        cp.int32(cp.int32(-l + S1) - 1),
                    )
                    temp += inp[iPixelPos] * kernel[coefPos]

        out[oPixelPos] = temp


def _numba_convolve_2d_signature(ty):
    return void(
        ty[:, :],  # inp
        int64,  # inpW
        int64,  # inpH
        ty[:, :],  # kernel
        int64,  # S0
        int64,  # S1 - only used by non-squares
        ty[:, :],  # out
        int64,  # outW
        int64,  # outH
        int64,  # pick
    )


# Custom Cupy raw kernel implementing upsample, filter, downsample operation
# Matthew Nicely - mnicely@nvidia.com
loaded_from_source = Template(
    """
$header

extern "C" {
    __global__ void _cupy_correlate_2d(
            const ${datatype} * __restrict__ inp,
            const int inpW,
            const int inpH,
            const ${datatype} * __restrict__ kernel,
            const int kerW,
            const int kerH,
            const int S0,
            const int S1,
            ${datatype} * __restrict__ out,
            const int outW,
            const int outH,
            const int pick) {

        const int ty {
            static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) };
        const int tx {
            static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) };

        int i {};
        if ( pick != 3 ) {
            i = tx + S0;
        } else {
            i = tx + S1;
        }
        int j { ty + S0 };

        int2 oPixelPos {tx, ty};
        if ( (tx < outH) && (ty < outW) ) {
            ${datatype} temp {};

            // Odd
            if ( pick == 1) {
                for (int k = -S0; k < (S0 + 1); k++){
                    for (int l = -S0; l < (S0 + 1); l++) {
                        int2 iPixelPos {(i + k), (j + l)};
                        int2 coefPos {(k + S0), (l + S0)};
                        temp += inp[iPixelPos.x * inpW + iPixelPos.y] *
                            kernel[coefPos.x * kerW + coefPos.y];
                    }
                }

            // Even
            } else if (pick == 2) {
                for (int k = -S0; k < S0; k++){
                    for (int l = -S0; l < S0; l++) {
                        int2 iPixelPos {(i + k), (j + l)}; // iPixelPos[1], [0]
                        int2 coefPos {(k + S0), (l + S0)};
                        temp += inp[iPixelPos.x * inpW + iPixelPos.y] *
                            kernel[coefPos.x * kerW + coefPos.y];
                    }
                }

            // Non-squares
            } else {
                for (int k = 0; k < S0; k++){
                    for (int l = 0; l < S1; l++) {
                        int2 iPixelPos {(i + k - S1), (j + l - S0)};
                        int2 coefPos {k, l};
                        temp += inp[iPixelPos.x * inpW + iPixelPos.y] *
                            kernel[coefPos.x * kerH + coefPos.y];
                    }
                }
            }
            out[oPixelPos.x * outW + oPixelPos.y] = temp;
        }
    }

    __global__ void _cupy_convolve_2d(
            const ${datatype} * __restrict__ inp,
            const int inpW,
            const int inpH,
            const ${datatype} * __restrict__ kernel,
            const int kerW,
            const int kerH,
            const int S0,
            const int S1,
            ${datatype} * __restrict__ out,
            const int outW,
            const int outH,
            const int pick) {

        const int ty {
            static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) };
        const int tx {
            static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) };

        int i {};
        if ( pick != 3 ) {
            i = tx + S0;
        } else {
            i = tx + S1;
        }
        int j { ty + S0 };

        int2 oPixelPos {tx, ty};
        if ( (tx < outH) && (ty < outW) ) {
            ${datatype} temp {};

            // Odd kernel
            if ( pick == 1) {
                for (int k = -S0; k < (S0 + 1); k++){
                    for (int l = -S0; l < (S0 + 1); l++) {
                        int2 iPixelPos {(i + k), (j + l)};
                        int2 coefPos {(-k + S0), (-l + S0)};
                        temp += inp[iPixelPos.x * inpW + iPixelPos.y] *
                            kernel[coefPos.x * kerW + coefPos.y];
                    }
                }
            // Even kernel
            } else if (pick == 2) {
                for (int k = -S0; k < S0; k++){
                    for (int l = -S0; l < S0; l++) {
                        int2 iPixelPos {(i + k), (j + l)};
                        int2 coefPos {(-k + S0 - 1), (-l + S0 - 1)};
                        temp += inp[iPixelPos.x * inpW + iPixelPos.y] *
                            kernel[coefPos.x * kerW + coefPos.y];
                    }
                }

            // Non-squares kernel
            } else {
                for (int k = 0; k < S0; k++){
                    for (int l = 0; l < S1; l++) {
                        int2 iPixelPos {(i + k - S1), (j + l - S0)};
                        int2 coefPos {(-k + S0 - 1), (-l + S1 - 1)};
                        temp += inp[iPixelPos.x * inpW + iPixelPos.y] *
                            kernel[coefPos.x * kerH + coefPos.y];
                    }
                }
            }
            out[oPixelPos.x * outW + oPixelPos.y] = temp;
        }
    }

    __global__ void _cupy_correlate(
            const ${datatype} * __restrict__ inp,
            const int inpW,
            const ${datatype} * __restrict__ kernel,
            const int kerW,
            const int mode,
            const bool swapped_inputs,
            ${datatype} * __restrict__ out,
            const int outW) {

        const int tx {
            static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
        const int stride { static_cast<int>( blockDim.x * gridDim.x ) };

        for ( int tid = tx; tid < outW; tid += stride ) {
            ${datatype} temp {};

            if ( mode == 0 ) {  // Valid
                if ( tid >= 0 && tid < inpW ) {
                    for ( int j = 0; j < kerW; j++ ) {
                        temp += inp[tid + j] * kernel[j];
                    }
                }
            } else if ( mode == 1 ) {   // Same
                const int P1 { kerW / 2 };
                int start {};
                if ( !swapped_inputs ) {
                    start = 0 - P1 + tid;
                } else {
                    start = ( ( inpW - 1 ) / 2 ) - ( kerW - 1 ) + tid;
                }
                for ( int j = 0; j < kerW; j++ ) {
                    if ( ( start + j >= 0 ) && ( start + j < inpW ) ) {
                        temp += inp[start + j] * kernel[j];
                    }
                }
            } else {    // Full
                const int P1 { kerW - 1 };
                int start { 0 - P1 + tid };
                for ( int j = 0; j < kerW; j++ ) {
                    if ( ( start + j >= 0 ) && ( start + j < inpW ) ) {
                        temp += inp[start + j] * kernel[j];
                    }
                }
            }

            if (swapped_inputs) {
                out[outW - tid - 1] = temp; // TODO: Move to shared memory
            } else {
                out[tid] = temp;
            }
        }
    }

    __global__ void _cupy_convolve(
            const ${datatype} * __restrict__ inp,
            const int inpW,
            const ${datatype} * __restrict__ kernel,
            const int kerW,
            const int mode,
            const bool swapped_inputs,
            ${datatype} * __restrict__ out,
            const int outW) {

        const int tx {
            static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
        const int stride { static_cast<int>( blockDim.x * gridDim.x ) };

        for ( int tid = tx; tid < outW; tid += stride ) {

            ${datatype} temp {};

            if ( mode == 0 ) {  // Valid
                if ( tid >= 0 && tid < inpW ) {
                    for ( int j = 0; j < kerW; j++ ) {
                        temp += inp[tid + j] * kernel[( kerW - 1 ) - j];
                    }
                }
            } else if ( mode == 1 ) {   // Same
                const int P1 { kerW / 2 };
                int start {};
                if ( !swapped_inputs ) {
                    start = 0 - P1 + tid;
                } else {
                    start = ( ( inpW - 1 ) / 2 ) - ( kerW - 1 ) + tid;
                }
                for ( int j = 0; j < kerW; j++ ) {
                    if ( ( start + j >= 0 ) && ( start + j < inpW ) ) {
                        temp += inp[start + j] * kernel[( kerW - 1 ) - j];
                    }
                }
            } else {    // Full
                const int P1 { kerW - 1 };
                int start { 0 - P1 + tid };
                for ( int j = 0; j < kerW; j++ ) {
                    if ( ( start + j >= 0 ) && ( start + j < inpW ) ) {
                        temp += inp[start + j] * kernel[( kerW - 1 ) - j];
                    }
                }
            }

            out[tid] = temp;
        }
    }
}
"""
)


class _cupy_convolve_wrapper(object):
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
        self, d_inp, d_kernel, mode, swapped_inputs, out,
    ):

        kernel_args = (
            d_inp,
            d_inp.shape[0],
            d_kernel,
            d_kernel.shape[0],
            mode,
            swapped_inputs,
            out,
            out.shape[0],
        )

        self.stream.use()
        self.kernel(self.grid, self.block, kernel_args)


class _cupy_convolve_2d_wrapper(object):
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
        self, d_inp, paddedW, paddedH, d_kernel, S0, S1, out, outW, outH, pick,
    ):

        kernel_args = (
            d_inp,
            paddedW,
            paddedH,
            d_kernel,
            d_kernel.shape[0],
            d_kernel.shape[1],
            S0,
            S1,
            out,
            outW,
            outH,
            pick,
        )

        self.stream.use()
        self.kernel(self.grid, self.block, kernel_args)


def _get_backend_kernel(
    dtype, grid, block, stream, use_numba, k_type,
):

    if not use_numba:
        kernel = _cupy_kernel_cache[(dtype.name, k_type)]
        if kernel:
            if k_type == GPUKernel.CONVOLVE or k_type == GPUKernel.CORRELATE:
                return _cupy_convolve_wrapper(grid, block, stream, kernel)
            elif (
                k_type == GPUKernel.CONVOLVE2D
                or k_type == GPUKernel.CORRELATE2D
            ):
                return _cupy_convolve_2d_wrapper(grid, block, stream, kernel)
            else:
                raise NotImplementedError(
                    "No CuPY kernel found for k_type {}, datatype {}".format(
                        k_type, dtype
                    )
                )
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
        if (str(numba_type), k_type) in _cupy_kernel_cache:
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
        if k_type == GPUKernel.CORRELATE:
            _cupy_kernel_cache[
                (str(numba_type), GPUKernel.CORRELATE)
            ] = module.get_function("_cupy_correlate")
        elif k_type == GPUKernel.CONVOLVE:
            _cupy_kernel_cache[
                (str(numba_type), GPUKernel.CONVOLVE)
            ] = module.get_function("_cupy_convolve")
        elif k_type == GPUKernel.CORRELATE2D:
            _cupy_kernel_cache[
                (str(numba_type), GPUKernel.CORRELATE2D)
            ] = module.get_function("_cupy_correlate_2d")
        elif k_type == GPUKernel.CONVOLVE2D:
            _cupy_kernel_cache[
                (str(numba_type), GPUKernel.CONVOLVE2D)
            ] = module.get_function("_cupy_convolve_2d")
        else:
            raise NotImplementedError(
                "No kernel found for k_type {}, datatype {}".format(
                    k_type, str(numba_type)
                )
            )

    else:
        if (str(numba_type), k_type) in _numba_kernel_cache:
            return
        # JIT compile the numba kernels
        # k_type = 0/1 (correlate/convolve)
        sig = _numba_convolve_2d_signature(numba_type)
        if k_type == GPUKernel.CORRELATE2D:
            _numba_kernel_cache[(str(numba_type), k_type)] = cuda.jit(
                sig, fastmath=True
            )(_numba_correlate_2d)
        elif k_type == GPUKernel.CONVOLVE2D:
            _numba_kernel_cache[(str(numba_type), k_type)] = cuda.jit(
                sig, fastmath=True
            )(_numba_convolve_2d)
        elif k_type == GPUKernel.CONVOLVE or k_type == GPUKernel.CORRELATE:
            return
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
            np.int32
            np.int64
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
            GPUKernel.CORRELATE
            GPUKernel.CONVOLVE
            GPUKernel.CORRELATE2D
            GPUKernel.CONVOLVE2D
        Examples
    ----------
    To precompile all kernels in this unit
    >>> import cusignal
    >>> from cusignal._upfirdn import GPUBackend, GPUKernel
    >>> cusignal._signaltools.precompile_kernels()

    To precompile a specific NumPy datatype, CuPy backend, and kernel type
    >>> cusignal._signaltools.precompile_kernels( [np.float64],
        [GPUBackend.CUPY], [GPUKernel.CORRELATE],)


    To precompile a specific NumPy datatype and kernel type,
    but both Numba and CuPY variations
    >>> cusignal._signaltools.precompile_kernels( dtype=[np.float64],
        k_type=[GPUKernel.CORRELATE],)
    """
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


def _convolve_gpu(
    inp, out, ker, mode, use_convolve, swapped_inputs, cp_stream,
):

    d_inp = cp.array(inp)
    d_kernel = cp.array(ker)

    device_id = cp.cuda.Device()
    numSM = device_id.attributes["MultiProcessorCount"]

    threadsperblock = 256
    blockspergrid = numSM * 20

    if use_convolve:
        _populate_kernel_cache(out.dtype.type, False, GPUKernel.CONVOLVE)
        kernel = _get_backend_kernel(
            out.dtype,
            blockspergrid,
            threadsperblock,
            cp_stream,
            False,
            GPUKernel.CONVOLVE,
        )
    else:
        _populate_kernel_cache(out.dtype.type, False, GPUKernel.CORRELATE)
        kernel = _get_backend_kernel(
            out.dtype,
            blockspergrid,
            threadsperblock,
            cp_stream,
            False,
            GPUKernel.CORRELATE,
        )

    kernel(d_inp, d_kernel, mode, swapped_inputs, out)

    return out


def _convolve2d_gpu(
    inp,
    out,
    ker,
    mode,
    boundary,
    use_convolve,
    fillvalue,
    cp_stream,
    use_numba,
):

    if (boundary != PAD) and (boundary != REFLECT) and (boundary != CIRCULAR):
        raise Exception("Invalid boundary flag")

    S = np.zeros(2, dtype=int)

    # If kernel is square and odd
    if ker.shape[0] == ker.shape[1]:  # square
        if ker.shape[0] % 2 == 1:  # odd
            pick = 1
            S[0] = (ker.shape[0] - 1) // 2
            if mode == 2:  # full
                P1 = P2 = P3 = P4 = S[0] * 2
            else:  # same/valid
                P1 = P2 = P3 = P4 = S[0]
        else:  # even
            pick = 2
            S[0] = ker.shape[0] // 2
            if mode == 2:  # full
                P1 = P2 = P3 = P4 = S[0] * 2 - 1
            else:  # same/valid
                if use_convolve:
                    P1 = P2 = P3 = P4 = S[0]
                else:
                    P1 = P3 = S[0] - 1
                    P2 = P4 = S[0]
    else:  # Non-square
        pick = 3
        S[0] = ker.shape[0]
        S[1] = ker.shape[1]
        if mode == 2:  # full
            P1 = S[0] - 1
            P2 = S[0] - 1
            P3 = S[1] - 1
            P4 = S[1] - 1
        else:  # same/valid
            if use_convolve:
                P1 = S[0] // 2
                P2 = S[0] // 2 if (S[0] % 2) else S[0] // 2 - 1
                P3 = S[1] // 2
                P4 = S[1] // 2 if (S[1] % 2) else S[1] // 2 - 1
            else:
                P1 = S[0] // 2 if (S[0] % 2) else S[0] // 2 - 1
                P2 = S[0] // 2
                P3 = S[1] // 2 if (S[1] % 2) else S[1] // 2 - 1
                P4 = S[1] // 2

    if mode == 1:  # SAME
        pad = ((P1, P2), (P3, P4))  # 4x5
        if boundary == REFLECT:
            inp = cp.pad(inp, pad, "symmetric")
        if boundary == CIRCULAR:
            inp = cp.pad(inp, pad, "wrap")
        if boundary == PAD:
            inp = cp.pad(inp, pad, "constant", constant_values=(fillvalue))

    if mode == 2:  # FULL
        pad = ((P1, P2), (P3, P4))
        if boundary == REFLECT:
            inp = cp.pad(inp, pad, "symmetric")
        if boundary == CIRCULAR:
            inp = cp.pad(inp, pad, "wrap")
        if boundary == PAD:
            inp = cp.pad(inp, pad, "constant", constant_values=(fillvalue))

    paddedW = inp.shape[1]
    paddedH = inp.shape[0]

    outW = out.shape[1]
    outH = out.shape[0]

    d_inp = cp.array(inp)
    d_kernel = cp.array(ker)

    threadsperblock = (16, 16)
    blockspergrid = (
        _iDivUp(outW, threadsperblock[0]),
        _iDivUp(outH, threadsperblock[1]),
    )

    if use_convolve:
        _populate_kernel_cache(out.dtype.type, use_numba, GPUKernel.CONVOLVE2D)
        kernel = _get_backend_kernel(
            out.dtype,
            blockspergrid,
            threadsperblock,
            cp_stream,
            use_numba,
            GPUKernel.CONVOLVE2D,
        )
    else:
        _populate_kernel_cache(
            out.dtype.type, use_numba, GPUKernel.CORRELATE2D
        )
        kernel = _get_backend_kernel(
            out.dtype,
            blockspergrid,
            threadsperblock,
            cp_stream,
            use_numba,
            GPUKernel.CORRELATE2D,
        )

    kernel(
        d_inp, paddedW, paddedH, d_kernel, S[0], S[1], out, outW, outH, pick
    )
    return out


def _convolve(
    in1,
    in2,
    use_convolve,
    swapped_inputs,
    mode,
    cp_stream=cp.cuda.stream.Stream(null=True),
):

    val = _valfrommode(mode)

    # Promote inputs
    promType = cp.promote_types(in1.dtype, in2.dtype)
    in1 = in1.astype(promType)
    in2 = in2.astype(promType)

    # Create empty array to hold number of aout dimensions
    out_dimens = np.empty(in1.ndim, np.int)
    if val == VALID:
        for i in range(in1.ndim):
            out_dimens[i] = (
                max(in1.shape[i], in2.shape[i])
                - min(in1.shape[i], in2.shape[i])
                + 1
            )
            if out_dimens[i] < 0:
                raise Exception(
                    "no part of the output is valid, use option 1 (same) or 2 \
                     (full) for third argument"
                )
    elif val == SAME:
        for i in range(in1.ndim):
            if not swapped_inputs:
                out_dimens[i] = in1.shape[i]  # Per scipy docs
            else:
                out_dimens[i] = min(in1.shape[i], in2.shape[i])
    elif val == FULL:
        for i in range(in1.ndim):
            out_dimens[i] = in1.shape[i] + in2.shape[i] - 1
    else:
        raise Exception("mode must be 0 (valid), 1 (same), or 2 (full)")

    # Create empty array out on GPU
    out = cp.empty(out_dimens.tolist(), in1.dtype)

    out = _convolve_gpu(
        in1, out, in2, val, use_convolve, swapped_inputs, cp_stream=cp_stream,
    )

    return out


def _convolve2d(
    in1,
    in2,
    use_convolve,
    mode="full",
    boundary="fill",
    fillvalue=0,
    cp_stream=cp.cuda.stream.Stream(null=True),
    use_numba=False,
):

    val = _valfrommode(mode)
    bval = _bvalfromboundary(boundary)

    # Promote inputs
    promType = cp.promote_types(in1.dtype, in2.dtype)
    in1 = in1.astype(promType)
    in2 = in2.astype(promType)

    if (bval != PAD) and (bval != REFLECT) and (bval != CIRCULAR):
        raise Exception("Incorrect boundary value.")

    if (bval == PAD) and (fillvalue is not None):
        fill = np.array(fillvalue, in1.dtype)
        if fill is None:
            raise Exception("fill must no be None.")
        if fill.size != 1:
            if fill.size == 0:
                raise Exception("`fillvalue` cannot be an empty array.")
            raise Exception(
                "`fillvalue` must be scalar or an array with one element"
            )
    else:
        fill = np.zeros(1, in1.dtype)
        if fill is None:
            raise Exception("Unable to create fill array")

    # Create empty array to hold number of aout dimensions
    out_dimens = np.empty(in1.ndim, np.int)
    if val == VALID:
        for i in range(in1.ndim):
            out_dimens[i] = in1.shape[i] - in2.shape[i] + 1
            if out_dimens[i] < 0:
                raise Exception(
                    "no part of the output is valid, use option 1 (same) or 2 \
                     (full) for third argument"
                )
    elif val == SAME:
        for i in range(in1.ndim):
            out_dimens[i] = in1.shape[i]
    elif val == FULL:
        for i in range(in1.ndim):
            out_dimens[i] = in1.shape[i] + in2.shape[i] - 1
    else:
        raise Exception("mode must be 0 (valid), 1 (same), or 2 (full)")

    # Create empty array out on GPU
    out = cp.empty(out_dimens.tolist(), in1.dtype)

    out = _convolve2d_gpu(
        in1,
        out,
        in2,
        val,
        bval,
        use_convolve,
        fill,
        cp_stream=cp_stream,
        use_numba=use_numba,
    )

    return out


def _design_resample_poly(up, down, window):
    """
    Design a prototype FIR low-pass filter using the window method
    for use in polyphase rational resampling.

    Parameters
    ----------
    up : int
        The upsampling factor.
    down : int
        The downsampling factor.
    window : string or tuple
        Desired window to use to design the low-pass filter.
        See below for details.

    Returns
    -------
    h : array
        The computed FIR filter coefficients.

    See Also
    --------
    resample_poly : Resample up or down using the polyphase method.

    Notes
    -----
    The argument `window` specifies the FIR low-pass filter design.
    The functions `scipy.signal.get_window` and `scipy.signal.firwin`
    are called to generate the appropriate filter coefficients.

    The returned array of coefficients will always be of data type
    `complex128` to maintain precision. For use in lower-precision
    filter operations, this array should be converted to the desired
    data type before providing it to `cusignal.resample_poly`.

    """

    # Determine our up and down factors
    # Use a rational approimation to save computation time on really long
    # signals
    g_ = gcd(up, down)
    up //= g_
    down //= g_

    # Design a linear-phase low-pass FIR filter
    max_rate = max(up, down)
    f_c = 1.0 / max_rate  # cutoff of FIR filter (rel. to Nyquist)

    # reasonable cutoff for our sinc-like function
    half_len = 10 * max_rate

    h = firwin(2 * half_len + 1, f_c, window=window)
    return h
