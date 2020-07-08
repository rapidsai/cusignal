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
import warnings

from enum import Enum
from pathlib import Path

from ._caches import _cupy_kernel_cache

# Display FutureWarnings only once per module
warnings.simplefilter("once", FutureWarning)


class GPUKernel(Enum):
    CORRELATE = "correlate"
    CONVOLVE = "convolve"
    CORRELATE2D = "correlate2d"
    CONVOLVE2D = "convolve2d"
    LOMBSCARGLE = "lombscargle"
    UNPACK = "unpack"
    PACK = "pack"
    SOSFILT = "sosfilt"
    UPFIRDN = "upfirdn"
    UPFIRDN2D = "upfirdn2d"


_SUPPORTED_TYPES_CONVOLVE = [
    "int32",
    "int64",
    "float32",
    "float64",
    "complex64",
    "complex128",
]
_SUPPORTED_TYPES_LOMBSCARGLE = ["float32", "float64"]
_SUPPORTED_TYPES_READER = [
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "float32",
    "float64",
    "complex64",
    "complex128",
]
_SUPPORTED_TYPES_SOSFILT = ["float32", "float64"]
_SUPPORTED_TYPES_UPFIRDN = ["float32", "float64", "complex64", "complex128"]


def _get_supported_types(k_type):

    if (
        k_type == GPUKernel.CORRELATE
        or k_type == GPUKernel.CONVOLVE
        or k_type == GPUKernel.CORRELATE2D
        or k_type == GPUKernel.CONVOLVE2D
    ):
        SUPPORTED_TYPES = _SUPPORTED_TYPES_CONVOLVE

    elif k_type == GPUKernel.LOMBSCARGLE:
        SUPPORTED_TYPES = _SUPPORTED_TYPES_LOMBSCARGLE

    elif k_type == GPUKernel.UNPACK or k_type == GPUKernel.PACK:
        SUPPORTED_TYPES = _SUPPORTED_TYPES_READER

    elif k_type == GPUKernel.SOSFILT:
        SUPPORTED_TYPES = _SUPPORTED_TYPES_SOSFILT

    elif k_type == GPUKernel.UPFIRDN or k_type == GPUKernel.UPFIRDN2D:
        SUPPORTED_TYPES = _SUPPORTED_TYPES_UPFIRDN

    else:
        raise ValueError("Support not found for '{}'".format(k_type.value))

    return SUPPORTED_TYPES


def _validate_input(dtype, k_type):

    k_type = list([k_type]) if k_type else list(GPUKernel)

    for k in k_type:

        # Point to types allowed for kernel
        SUPPORTED_TYPES = _get_supported_types(k)

        d = list(dtype) if dtype else SUPPORTED_TYPES

        for np_type in d:

            _populate_kernel_cache(np_type, k)


def _populate_kernel_cache(np_type, k_type):

    SUPPORTED_TYPES = _get_supported_types(k_type)

    if np_type not in SUPPORTED_TYPES:
        raise ValueError(
            "Datatype {} not found for '{}'".format(np_type, k_type.value)
        )

    if (str(np_type), k_type.value) in _cupy_kernel_cache:
        return

    mod_path = Path(__file__).parent
    relative_path = '..'

    dir = str((mod_path / relative_path).resolve())
    print(dir)

    if k_type == GPUKernel.CORRELATE:
        module = cp.RawModule(path=dir + "/convolution/_convolution.fatbin",)
        _cupy_kernel_cache[(str(np_type), k_type.value)] = module.get_function(
            "_cupy_correlate_" + str(np_type)
        )

    elif k_type == GPUKernel.CONVOLVE:
        module = cp.RawModule(path=dir + "/convolution/_convolution.fatbin",)
        _cupy_kernel_cache[(str(np_type), k_type.value)] = module.get_function(
            "_cupy_convolve_" + str(np_type)
        )

    elif k_type == GPUKernel.CORRELATE2D:
        module = cp.RawModule(path=dir + "/convolution/_convolution.fatbin",)
        _cupy_kernel_cache[(str(np_type), k_type.value)] = module.get_function(
            "_cupy_correlate2D_" + str(np_type)
        )

    elif k_type == GPUKernel.CONVOLVE2D:
        module = cp.RawModule(path=dir + "/convolution/_convolution.fatbin",)
        _cupy_kernel_cache[(str(np_type), k_type.value)] = module.get_function(
            "_cupy_convolve2D_" + str(np_type)
        )

    elif k_type == GPUKernel.LOMBSCARGLE:
        module = cp.RawModule(
            path=dir + "/spectral_analysis/_spectral.fatbin",
        )
        _cupy_kernel_cache[(str(np_type), k_type.value)] = module.get_function(
            "_cupy_lombscargle_" + str(np_type)
        )

    elif k_type == GPUKernel.UNPACK:
        module = cp.RawModule(path=dir + "/io/_reader.fatbin",)
        _cupy_kernel_cache[(str(np_type), k_type.value)] = module.get_function(
            "_cupy_unpack_" + str(np_type)
        )

    elif k_type == GPUKernel.PACK:
        module = cp.RawModule(path=dir + "/io/_writer.fatbin",)
        _cupy_kernel_cache[(str(np_type), k_type.value)] = module.get_function(
            "_cupy_pack_" + str(np_type)
        )

    elif k_type == GPUKernel.SOSFILT:
        module = cp.RawModule(path=dir + "/filtering/_sosfilt.fatbin",)
        _cupy_kernel_cache[(str(np_type), k_type.value)] = module.get_function(
            "_cupy_sosfilt_" + str(np_type)
        )

    elif k_type == GPUKernel.UPFIRDN:
        module = cp.RawModule(path=dir + "/filtering/_upfirdn.fatbin",)
        _cupy_kernel_cache[(str(np_type), k_type.value)] = module.get_function(
            "_cupy_upfirdn1D_" + str(np_type)
        )

    elif k_type == GPUKernel.UPFIRDN2D:
        module = cp.RawModule(path=dir + "/filtering/_upfirdn.fatbin",)
        _cupy_kernel_cache[(str(np_type), k_type.value)] = module.get_function(
            "_cupy_upfirdn2D_" + str(np_type)
        )

    else:
        raise NotImplementedError(
            "No kernel found for k_type {}, datatype {}".format(
                k_type, str(np_type)
            )
        )


def precompile_kernels(k_type=None, dtype=None):
    r"""
    Precompile GPU kernels for later use.

    Note: If a specified kernel + data type combination at runtime
    does not match any precompiled kernels, it will be compile at
    first call (if kernel and data type combination exist)

    Parameters
    ----------
    k_type : {str}, optional
        Which GPU kernel to compile for. If not specified,
        all supported kernels will be precompiled.
            'correlate'
            'convolve'
            'correlate2d'
            'convolve2d'
            'lombscargle'
            'upfirdn'
            'upfirdn2d'
    dtype : dtype or list of dtype, optional
        Data types for which kernels should be precompiled. If not
        specified, all supported data types will be precompiled.
            'correlate'
            'convolve'
            'correlate2d'
            'convolve2d'
            {
                int32
                int64
                float32
                float64
                complex64
                complex128
            }
            'lombscargle'
            {
                float32
                float64
            }
            'upfirdn'
            'upfirdn2d'
            {
                float32
                float64
                complex64
                complex128
            }

    Examples
    ----------
    To precompile all kernels
    >>> import cusignal
    >>> cusignal.precompile_kernels()

    To precompile a specific kernel and dtype [list of dtype],
    >>> cusignal.precompile_kernels('convolve2d', [np.float32, np.float64])

    To precompile a specific kernel and all data types
    >>> cusignal.precompile_kernels('convolve2d')

    To precompile a specific data type and all kernels
    >>> cusignal.precompile_kernels(dtype=['float64'])

    To precompile a multiple kernels
    >>> cusignal.precompile_kernels('convolve2d', [np.float64])
    >>> cusignal.precompile_kernels('correlate', [np.float64])
    """

    warnings.warn(
        "precompile_kernels() will be removed in a later release",
        FutureWarning,
        stacklevel=2,
    )

    if k_type is not None and not isinstance(k_type, str):
        raise TypeError(
            "k_type is type ({}), should be (string) - e.g {}".format(
                type(k_type), "'convolve2d'"
            )
        )
    elif k_type is not None:
        k_type = k_type.lower()
        try:
            k_type = GPUKernel(k_type)

        except ValueError:
            raise

    if dtype is not None and not hasattr(dtype, "__iter__"):
        raise TypeError(
            "dtype ({}) should be in list - e.g [np.float32,]".format(dtype)
        )
    else:
        _validate_input(dtype, k_type)
