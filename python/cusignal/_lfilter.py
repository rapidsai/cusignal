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
from numba import float32, float64
from string import Template


class GPUKernel(Enum):
    LFILTER = 0


class GPUBackend(Enum):
    CUPY = 0


# Numba type supported and corresponding C type
_SUPPORTED_TYPES = {
    np.float32: [float32, "float"],
    np.float64: [float64, "double"],
}

_cupy_kernel_cache = {}


# Custom Cupy raw kernel implementing upsample, filter, downsample operation
# Matthew Nicely - mnicely@nvidia.com
loaded_from_source = Template(
    """
extern "C" {
    __global__ void _cupy_lfilter(
            const int x_len,
            const int a_len,
            const ${datatype} * __restrict__ x,
            const ${datatype} * __restrict__ a,
            const ${datatype} * __restrict__ b,
            ${datatype} * __restrict__ out) {

        for ( int tid = 0; tid < x_len; tid++) {

            ${datatype} isw {};
            ${datatype} wos {};

            // Create input_signal_windows
            if( tid > ( a_len ) ) {
                for ( int i = 0; i < a_len; i++ ) {
                    isw += x[tid - i] * b[i];
                    wos += out[tid - i] * a[i];
                }
            } else {
                for ( int i = 0; i <= tid; i++ ) {
                    isw += x[tid - i] * b[i];
                    wos += out[tid - i] * a[i];
                }
            }

            isw -= wos;

            out[tid] = isw / a[0];
        }
    }
}
"""
)


class _cupy_lfilter_wrapper(object):
    def __init__(self, grid, block, stream, kernel):
        if isinstance(grid, int):
            grid = (grid,)
        if isinstance(block, int):
            block = (block,)

        self.grid = grid
        self.block = block
        self.stream = stream
        self.kernel = kernel

    def __call__(self, b, a, x, out):

        kernel_args = (
            x.shape[0],
            a.shape[0],
            x,
            a,
            b,
            out,
        )

        self.stream.use()
        self.kernel(self.grid, self.block, kernel_args)


def _get_backend_kernel(
    dtype, grid, block, stream, use_numba, k_type,
):

    if not use_numba:
        kernel = _cupy_kernel_cache[(dtype.name, k_type)]
        if kernel:
            if k_type == GPUKernel.LFILTER:
                return _cupy_lfilter_wrapper(grid, block, stream, kernel)
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
        src = loaded_from_source.substitute(datatype=c_type)
        module = cp.RawModule(
            code=src, options=("-std=c++11", "-use_fast_math")
        )
        if k_type == GPUKernel.LFILTER:
            _cupy_kernel_cache[
                (str(numba_type), GPUKernel.LFILTER)
            ] = module.get_function("_cupy_lfilter")
        else:
            raise NotImplementedError(
                "No kernel found for k_type {}, datatype {}".format(
                    k_type, str(numba_type)
                )
            )
    else:
        raise NotImplementedError(
            "Numba kernel not implemented for k_type {}".format(k_type)
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
    backend : GPUBackend, optional
        Which GPU backend to precompile for. If not specified,
        all supported backends will be precompiled.
        Specific to this unit
            GPUBackend.CUPY
    k_type : GPUKernel, optional
        Which GPU kernel to compile for. If not specified,
        all supported kernels will be precompiled.
        Specific to this unit
            GPUKernel.LFILTER
        Examples
    ----------
    To precompile all kernels in this unit
    >>> import cusignal
    >>> from cusignal._upfirdn import GPUBackend, GPUKernel
    >>> cusignal._lfilter.precompile_kernels()

    To precompile a specific NumPy datatype, CuPy backend, and kernel type
    >>> cusignal._lfilter.precompile_kernels( [np.float64],
        [GPUBackend.CUPY], [GPUKernel.LFILTER],)


    To precompile a specific NumPy datatype and kernel type,
    but both Numba and CuPY variations
    >>> cusignal._lfilter.precompile_kernels( dtype=[np.float64],
        k_type=[GPUKernel.LFILTER],)
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


def _lfilter_gpu(b, a, x, clamp, cp_stream):

    out = cp.zeros_like(x)

    threadsperblock = 1
    blockspergrid = 1

    _populate_kernel_cache(out.dtype.type, False, GPUKernel.LFILTER)
    kernel = _get_backend_kernel(
        out.dtype,
        blockspergrid,
        threadsperblock,
        cp_stream,
        False,
        GPUKernel.LFILTER,
    )

    kernel(b, a, x, out)

    # Turn on in a different PR
    # cp_stream.synchronize()

    return out
