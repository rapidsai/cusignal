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

from numba import cuda, int32, int64, float32, float64, complex64, complex128
from enum import Enum

from ._caches import _cupy_kernel_cache, _numba_kernel_cache

from ._lfilter_kernels import _cupy_lfilter_src
from ._signaltools_kernels import (
    _cupy_correlate_src,
    _cupy_convolve_src,
    _cupy_correlate_2d_src,
    _cupy_convolve_2d_src,
    _numba_correlate_2d,
    _numba_convolve_2d,
    _numba_convolve_2d_signature,
)
from ._spectral_kernels import (
    _cupy_lombscargle_src,
    _numba_lombscargle,
    _numba_lombscargle_signature,
)

from ._upfirdn_kernels import (
    _cupy_upfirdn_1d_src,
    _cupy_upfirdn_2d_src,
    _numba_upfirdn_1d,
    _numba_upfirdn_2d,
    _numba_upfirdn_1d_signature,
    _numba_upfirdn_2d_signature,
)

try:
    # Numba <= 0.49
    from numba.types.scalars import Complex
except ImportError:
    # Numba >= 0.49
    from numba.core.types.scalars import Complex


class GPUKernel(Enum):
    CORRELATE = 0
    CONVOLVE = 1
    CORRELATE2D = 2
    CONVOLVE2D = 3
    LFILTER = 4
    LOMBSCARGLE = 5
    UPFIRDN = 6
    UPFIRDN2D = 7


class GPUBackend(Enum):
    CUPY = 0
    NUMBA = 1


# Numba type supported and corresponding C type
_SUPPORTED_TYPES_CONVOLVE = {
    np.int32: [int32, "int"],
    np.int64: [int64, "long int"],
    np.float32: [float32, "float"],
    np.float64: [float64, "double"],
    np.complex64: [complex64, "complex<float>"],
    np.complex128: [complex128, "complex<double>"],
}

_SUPPORTED_TYPES_LFILTER = {
    np.float32: [float32, "float"],
    np.float64: [float64, "double"],
}

_SUPPORTED_TYPES_LOMBSCARGLE = {
    np.float32: [float32, "float"],
    np.float64: [float64, "double"],
}

_SUPPORTED_TYPES_UPFIRDN = {
    np.float32: [float32, "float"],
    np.float64: [float64, "double"],
    np.complex64: [complex64, "complex<float>"],
    np.complex128: [complex128, "complex<double>"],
}


# Use until functionality provided in Numba 0.50 available
def _stream_cupy_to_numba(cp_stream):
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


def _get_supported_types(k_type):
    if (
        k_type == GPUKernel.CORRELATE
        or k_type == GPUKernel.CONVOLVE
        or k_type == GPUKernel.CORRELATE2D
        or k_type == GPUKernel.CONVOLVE2D
    ):
        SUPPORTED_TYPES = _SUPPORTED_TYPES_CONVOLVE

    elif k_type == GPUKernel.LFILTER:
        SUPPORTED_TYPES = _SUPPORTED_TYPES_LFILTER

    elif k_type == GPUKernel.LOMBSCARGLE:
        SUPPORTED_TYPES = _SUPPORTED_TYPES_LOMBSCARGLE

    elif k_type == GPUKernel.UPFIRDN or k_type == GPUKernel.UPFIRDN2D:
        SUPPORTED_TYPES = _SUPPORTED_TYPES_UPFIRDN

    else:
        raise ValueError("Support not found for {}".format(k_type))

    return SUPPORTED_TYPES


def _validate_input(dtype, backend, k_type):

    k_list = [k_type]
    b_list = [backend]

    backend = list(b_list) if backend else list(GPUBackend)
    k_type = list(k_list) if k_type else list(GPUKernel)

    for b, k in itertools.product(backend, k_type):
        # Check if use_numba is support
        try:
            GPUBackend(b.value)

        except ValueError:
            raise

        # Check if use_numba is support
        try:
            GPUKernel(k)

        except ValueError:
            raise

        # Point to types allowed for kernel
        SUPPORTED_TYPES = _get_supported_types(k)

        d = list(dtype) if dtype else SUPPORTED_TYPES.keys()

        for np_type in d:

            try:
                numba_type, c_type = SUPPORTED_TYPES[np_type]

            except KeyError:
                raise KeyError(
                    "Datatype {} not found for {}".format(np_type, k)
                )

            _populate_kernel_cache(np_type, b.value, k)


def _populate_kernel_cache(np_type, use_numba, k_type):

    SUPPORTED_TYPES = _get_supported_types(k_type)

    numba_type, c_type = SUPPORTED_TYPES[np_type]

    if not use_numba:
        if (str(numba_type), k_type.value) in _cupy_kernel_cache:
            return

        # Instantiate the cupy kernel for this type and compile
        if isinstance(numba_type, Complex):
            header = "#include <cupy/complex.cuh>"
        else:
            header = ""

        if k_type == GPUKernel.CORRELATE:
            src = _cupy_correlate_src.substitute(
                datatype=c_type, header=header
            )
            module = cp.RawModule(
                code=src, options=("-std=c++11", "-use_fast_math")
            )
            _cupy_kernel_cache[
                (str(numba_type), k_type.value)
            ] = module.get_function("_cupy_correlate")

        elif k_type == GPUKernel.CONVOLVE:
            src = _cupy_convolve_src.substitute(datatype=c_type, header=header)
            module = cp.RawModule(
                code=src, options=("-std=c++11", "-use_fast_math")
            )
            _cupy_kernel_cache[
                (str(numba_type), k_type.value)
            ] = module.get_function("_cupy_convolve")

        elif k_type == GPUKernel.CORRELATE2D:
            src = _cupy_correlate_2d_src.substitute(
                datatype=c_type, header=header
            )
            module = cp.RawModule(
                code=src, options=("-std=c++11", "-use_fast_math")
            )
            _cupy_kernel_cache[
                (str(numba_type), k_type.value)
            ] = module.get_function("_cupy_correlate_2d")

        elif k_type == GPUKernel.CONVOLVE2D:
            src = _cupy_convolve_2d_src.substitute(
                datatype=c_type, header=header
            )
            module = cp.RawModule(
                code=src, options=("-std=c++11", "-use_fast_math")
            )
            _cupy_kernel_cache[
                (str(numba_type), k_type.value)
            ] = module.get_function("_cupy_convolve_2d")

        elif k_type == GPUKernel.LOMBSCARGLE:
            src = _cupy_lombscargle_src.substitute(
                datatype=c_type, header=header
            )
            module = cp.RawModule(
                code=src, options=("-std=c++11", "-use_fast_math")
            )
            _cupy_kernel_cache[
                (str(numba_type), k_type.value)
            ] = module.get_function("_cupy_lombscargle")
        elif k_type == GPUKernel.LFILTER:
            src = _cupy_lfilter_src.substitute(datatype=c_type, header=header)
            module = cp.RawModule(
                code=src, options=("-std=c++11", "-use_fast_math")
            )
            _cupy_kernel_cache[
                (str(numba_type), k_type.value)
            ] = module.get_function("_cupy_lfilter")
        elif k_type == GPUKernel.UPFIRDN:
            src = _cupy_upfirdn_1d_src.substitute(
                datatype=c_type, header=header
            )
            module = cp.RawModule(
                code=src, options=("-std=c++11", "-use_fast_math")
            )
            _cupy_kernel_cache[
                (str(numba_type), k_type.value)
            ] = module.get_function("_cupy_upfirdn_1d")
        elif k_type == GPUKernel.UPFIRDN2D:
            src = _cupy_upfirdn_2d_src.substitute(
                datatype=c_type, header=header
            )
            module = cp.RawModule(
                code=src, options=("-std=c++11", "-use_fast_math")
            )
            _cupy_kernel_cache[
                (str(numba_type), k_type.value)
            ] = module.get_function("_cupy_upfirdn_2d")

        else:
            raise NotImplementedError(
                "No kernel found for k_type {}, datatype {}".format(
                    k_type, str(numba_type)
                )
            )

    else:
        if (str(numba_type), k_type.value) in _numba_kernel_cache:
            return
        # JIT compile the numba kernels
        if k_type == GPUKernel.CONVOLVE or k_type == GPUKernel.CORRELATE:
            raise NotImplementedError(
                "{} Numba has no Numba Implementation".format(
                    k_type
                )
            )
        if k_type == GPUKernel.CORRELATE2D:
            sig = _numba_convolve_2d_signature(numba_type)
            _numba_kernel_cache[(str(numba_type), k_type.value)] = cuda.jit(
                sig, fastmath=True
            )(_numba_correlate_2d)

        elif k_type == GPUKernel.CONVOLVE2D:
            sig = _numba_convolve_2d_signature(numba_type)
            _numba_kernel_cache[(str(numba_type), k_type.value)] = cuda.jit(
                sig, fastmath=True
            )(_numba_convolve_2d)

        elif k_type == GPUKernel.LOMBSCARGLE:
            sig = _numba_lombscargle_signature(numba_type)
            _numba_kernel_cache[(str(numba_type), k_type.value)] = cuda.jit(
                sig, fastmath=True
            )(_numba_lombscargle)
        elif k_type == GPUKernel.LFILTER:
            raise NotImplementedError(
                "{} Numba has no Numba Implementation".format(
                    k_type
                )
            )
        elif k_type == GPUKernel.UPFIRDN:
            sig = _numba_upfirdn_1d_signature(numba_type)
            _numba_kernel_cache[(str(numba_type), k_type.value)] = cuda.jit(
                sig, fastmath=True
            )(_numba_upfirdn_1d)
        elif k_type == GPUKernel.UPFIRDN2D:
            sig = _numba_upfirdn_2d_signature(numba_type)
            _numba_kernel_cache[(str(numba_type), k_type.value)] = cuda.jit(
                sig, fastmath=True
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
            GPUKernel.CORRELATE
            GPUKernel.CONVOLVE
            GPUKernel.CORRELATE2D
            GPUKernel.CONVOLVE2D
            {
                np.int32
                np.int64
                np.float32
                np.float64
                np.complex64
                np.complex128
            }
            GPUKernel.LFILTER
            GPUKernel.LOMBSCARGLE
            {
                np.float32
                np.float64
            }
            GPUKernel.UPFIRDN
            GPUKernel.UPFIRDN2D
            {
                np.float32
                np.float64
                np.complex64
                np.complex128
            }
    backend : GPUBackend, optional
        Which GPU backend to precompile for. If not specified,
        all supported backends will be precompiled.
            GPUKernel.CORRELATE2D
            GPUKernel.CONVOLVE2D
            GPUKernel.UPFIRDN
            GPUKernel.UPFIRDN2D
            {
                GPUBackend.CUPY
                GPUBackend.NUMBA
            }

            GPUKernel.CORRELATE
            GPUKernel.CONVOLVE
            GPUKernel.LFILTER
            GPUKernel.LOMBSCARGLE
            {
                GPUBackend.CUPY
            }

    k_type : GPUKernel, optional
        Which GPU kernel to compile for. If not specified,
        all supported kernels will be precompiled.
            GPUKernel.CORRELATE
            GPUKernel.CONVOLVE
            GPUKernel.CORRELATE2D
            GPUKernel.CONVOLVE2D
            GPUKernel.LFILTER
            GPUKernel.LOMBSCARGLE
            GPUKernel.UPFIRDN
            GPUKernel.UPFIRDN2D
        Examples
    ----------
    To precompile all kernels in this unit
    >>> import cusignal
    >>> from cusignal._upfirdn import GPUBackend, GPUKernel
    >>> cusignal._lfilter.precompile_kernels()

    To precompile a specific NumPy data type [list of data types],
    CuPy backend, and kernel type
    >>> cusignal._lfilter.precompile_kernels( [np.float32, np.float64],
        GPUBackend.CUPY, GPUKernel.LFILTER,)


    To precompile a specific NumPy data type and kernel type,
    but both Numba and CuPY variations
    >>> cusignal._lfilter.precompile_kernels( dtype=[np.float64],
        k_type=GPUKernel.LFILTER,)
    """

    if dtype is not None and not hasattr(dtype, "__iter__"):
        raise TypeError(
            "dtype ({}) should be in list - e.g [np.float32,]".format(dtype)
        )

    elif backend is not None and not isinstance(backend, GPUBackend):
        raise TypeError(
            "backend is type ({}), should be (GPUBackend) - e.g {}".format(
                type(k_type), GPUBackend.CUPY
            )
        )

    elif k_type is not None and not isinstance(k_type, GPUKernel):
        raise TypeError(
            "k_type is type ({}), should be (GPUKernel) - e.g {}".format(
                type(k_type), GPUKernel.CORRELATE
            )
        )
    else:
        _validate_input(dtype, backend, k_type)
