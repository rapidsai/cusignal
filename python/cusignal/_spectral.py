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

from enum import Enum
from math import sin, cos, atan2
from string import Template
from numba import (
    cuda,
    float64,
    void,
)
import itertools
import cupy as cp
import numpy as np


class GPUBackend(Enum):
    CUPY = 0
    NUMBA = 1


_SUPPORTED_TYPES = {
    np.float64: [float64, "double"],
}


_numba_kernel_cache = {}
_cupy_kernel_cache = {}

# Numba type supported and corresponding C type


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


def _numba_lombscargle(x, y, freqs, pgram, y_dot):
    """
    _lombscargle(x, y, freqs)
    Computes the Lomb-Scargle periodogram.
    Parameters
    ----------
    x : array_like
        Sample times.
    y : array_like
        Measurement values (must be registered so the mean is zero).
    freqs : array_like
        Angular frequencies for output periodogram.
    Returns
    -------
    pgram : array_like
        Lomb-Scargle periodogram.
    Raises
    ------
    ValueError
        If the input arrays `x` and `y` do not have the same shape.
    See also
    --------
    lombscargle
    """

    F = cuda.grid(1)
    strideF = cuda.gridsize(1)

    if not y_dot[0]:
        yD = 1.0
    else:
        yD = 2.0 / y_dot[0]

    for i in range(F, freqs.shape[0], strideF):

        # Copy data to registers
        freq = freqs[i]

        xc = 0.0
        xs = 0.0
        cc = 0.0
        ss = 0.0
        cs = 0.0

        for j in range(x.shape[0]):

            c = cos(freq * x[j])
            s = sin(freq * x[j])

            xc += y[j] * c
            xs += y[j] * s
            cc += c * c
            ss += s * s
            cs += c * s

        tau = atan2(2.0 * cs, cc - ss) / (2.0 * freq)
        c_tau = cos(freq * tau)
        s_tau = sin(freq * tau)
        c_tau2 = c_tau * c_tau
        s_tau2 = s_tau * s_tau
        cs_tau = 2.0 * c_tau * s_tau

        pgram[i] = (
            0.5
            * (
                (
                    (c_tau * xc + s_tau * xs) ** 2
                    / (c_tau2 * cc + cs_tau * cs + s_tau2 * ss)
                )
                + (
                    (c_tau * xs - s_tau * xc) ** 2
                    / (c_tau2 * ss - cs_tau * cs + s_tau2 * cc)
                )
            )
        ) * yD


def _numba_lombscargle_signature(ty):
    return void(
        ty[:], ty[:], ty[:], ty[:], ty[:],  # x  # y  # freqs  # pgram  # y_dot
    )


# Custom Cupy raw kernel implementing lombscargle operation
# Matthew Nicely - mnicely@nvidia.com
loaded_from_source = Template(
    """
extern "C" {
    __global__ void _cupy_lombscargle(
            const int x_shape,
            const int freqs_shape,
            const ${datatype} * __restrict__ x,
            const ${datatype} * __restrict__ y,
            const ${datatype} * __restrict__ freqs,
            ${datatype} * __restrict__ pgram,
            const ${datatype} * __restrict__ y_dot
            ) {

        const int tx {
            static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
        const int stride { static_cast<int>( blockDim.x * gridDim.x ) };

        ${datatype} yD {};
        if ( y_dot[0] == 0 ) {
            yD = 1.0f;
        } else {
            yD = 2.0f / y_dot[0];
        }

        for ( int tid = tx; tid < freqs_shape; tid += stride ) {

            ${datatype} freq { freqs[tid] };

            ${datatype} xc {};
            ${datatype} xs {};
            ${datatype} cc {};
            ${datatype} ss {};
            ${datatype} cs {};
            ${datatype} c {};
            ${datatype} s {};

            for ( int j = 0; j < x_shape; j++ ) {
                c = cos( freq * x[j] );
                s = sin( freq * x[j] );

                xc += y[j] * c;
                xs += y[j] * s;
                cc += c * c;
                ss += s * s;
                cs += c * s;
            }

            ${datatype} tau { atan2( 2.0f * cs, cc - ss ) / ( 2.0f * freq ) };
            ${datatype} c_tau { cos(freq * tau) };
            ${datatype} s_tau { sin(freq * tau) };
            ${datatype} c_tau2 { c_tau * c_tau };
            ${datatype} s_tau2 { s_tau * s_tau };
            ${datatype} cs_tau { 2.0f * c_tau * s_tau };

            pgram[tid] = (
                0.5f * (
                   (
                       ( c_tau * xc + s_tau * xs )
                       * ( c_tau * xc + s_tau * xs )
                       / ( c_tau2 * cc + cs_tau * cs + s_tau2 * ss )
                    )
                   + (
                       ( c_tau * xs - s_tau * xc )
                       * ( c_tau * xs - s_tau * xc )
                       / ( c_tau2 * ss - cs_tau * cs + s_tau2 * cc )
                    )
                )
            ) * yD;
        }
    }
}
"""
)


class _cupy_lombscargle_wrapper(object):
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
        self, x, y, freqs, pgram, y_dot,
    ):

        kernel_args = (
            x.shape[0],
            freqs.shape[0],
            x,
            y,
            freqs,
            pgram,
            y_dot,
        )

        self.stream.use()
        self.kernel(self.grid, self.block, kernel_args)


def _get_backend_kernel(dtype, grid, block, stream, use_numba):

    if use_numba:
        nb_stream = stream_cupy_to_numba(stream)
        kernel = _numba_kernel_cache[(dtype.name)]

        if kernel:
            return kernel[grid, block, nb_stream]
    else:
        kernel = _cupy_kernel_cache[(dtype.name)]
        if kernel:
            return _cupy_lombscargle_wrapper(grid, block, stream, kernel)

    raise NotImplementedError(
        "No kernel found for datatype {}".format(dtype.name)
    )


def _populate_kernel_cache(np_type, use_numba):

    try:
        numba_type, c_type = _SUPPORTED_TYPES[np_type]

    except KeyError:
        raise Exception("No kernel found for datatype {}".format(np_type))

    # JIT compile the numba kernels
    if not use_numba:
        if str(numba_type) in _cupy_kernel_cache:
            return
        # Instantiate the cupy kernel for this type and compile
        src = loaded_from_source.substitute(datatype=c_type)
        module = cp.RawModule(
            code=src, options=("-std=c++11", "-use_fast_math")
        )
        _cupy_kernel_cache[(str(numba_type))] = module.get_function(
            "_cupy_lombscargle"
        )

    else:
        if str(numba_type) in _numba_kernel_cache:
            return
        # JIT compile the numba kernels
        sig = _numba_lombscargle_signature(numba_type)
        _numba_kernel_cache[(str(numba_type))] = cuda.jit(sig, fastmath=True)(
            _numba_lombscargle
        )


def precompile_kernels(dtype=None, backend=None):
    r"""
    Precompile GPU kernels for later use.

    Parameters
    ----------
    dtype : numpy datatype or list of datatypes, optional
        Data types for which kernels should be precompiled. If not
        specified, all supported data types will be precompiled.
    backend : GPUBackend, optional
        Which GPU backend to precompile for. If not specified,
        all supported backends will be precompiled.
    """
    dtype = list(dtype) if dtype else _SUPPORTED_TYPES.keys()
    backend = list(backend) if backend else list(GPUBackend)
    for d, b in itertools.product(dtype, backend):
        _populate_kernel_cache(d, b.value)


def _lombscargle(x, y, freqs, pgram, y_dot, cp_stream, use_numba):

    device_id = cp.cuda.Device()
    numSM = device_id.attributes["MultiProcessorCount"]
    threadsperblock = 256
    blockspergrid = numSM * 20

    print(pgram.dtype.type)

    _populate_kernel_cache(pgram.dtype.type, use_numba)

    kernel = _get_backend_kernel(
        pgram.dtype, blockspergrid, threadsperblock, cp_stream, use_numba,
    )

    kernel(x, y, freqs, pgram, y_dot)
