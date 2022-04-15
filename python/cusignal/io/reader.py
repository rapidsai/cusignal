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

import json
import re

import cupy as cp
import numpy as np

from ._reader_cuda import _unpack


# https://hackersandslackers.com/extract-data-from-complex-json-python/
def _extract_values(obj, key):
    """Pull all values of specified key from nested JSON."""
    arr = []

    def extract(obj, arr, key):
        """Recursively search for values of key in JSON tree."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    extract(v, arr, key)
                elif k == key:
                    arr.append(v)
        elif isinstance(obj, list):
            for item in obj:
                extract(item, arr, key)
        return arr

    results = extract(obj, arr, key)
    return results


def read_bin(file, buffer=None, dtype=cp.uint8, num_samples=None, offset=0):
    """
    Reads binary file into GPU memory.
    Can be used as a building blocks for custom unpack/pack
    data readers/writers.

    Parameters
    ----------
    file : str
        A string of filename to be read to GPU.
    buffer : ndarray, optional
        Pinned memory buffer to use when copying data from GPU.
    dtype : data-type, optional
        Any object that can be interpreted as a numpy data type.
    num_samples : int, optional
        Number of samples to be loaded to GPU. If set to 0,
        read in all samples.
    offset : int, optional
        In the file, array data starts at this offset.
        Since offset is measured in bytes, it should normally
        be a multiple of the byte-size of dtype.
    Returns
    -------
    out : ndarray
        An 1-dimensional array containing binary data.

    """

    # Get current stream, default or not.
    stream = cp.cuda.get_current_stream()

    # offset is measured in bytes
    offset *= cp.dtype(dtype).itemsize

    fp = np.memmap(file, mode="r", offset=offset, shape=num_samples, dtype=dtype)

    if buffer is not None:
        out = cp.empty(buffer.shape, buffer.dtype)

    if buffer is None:
        out = cp.asarray(fp)
    else:
        buffer[:] = fp[:]
        out.set(buffer)

    stream.synchronize()

    del fp

    return out


def unpack_bin(binary, dtype, endianness="L"):
    """
    Unpack binary file.
    If endianness is big-endian, it my be converted
    to little endian for NVIDIA GPU compatibility.

    Parameters
    ----------
    binary : ndarray
        The binary array to be unpack.
    dtype : data-type, optional
        Any object that can be interpreted as a numpy data type.
    endianness : {'L', 'B'}, optional
        Data set byte order

    Returns
    -------
    out : ndarray
        An 1-dimensional array containing unpacked binary data.

    """

    if endianness != "L" and endianness != "B" and endianness != "N":
        raise ValueError("'endianness' should be 'L' or 'B'")

    out = _unpack(binary, dtype, endianness)

    return out


def read_sigmf(data_file, meta_file=None, buffer=None, num_samples=None, offset=0):
    """
    Read and unpack binary file, with SigMF spec, to GPU memory.

    Parameters
    ----------
    data_file : str
        File contain sigmf data.
    meta_file : str, optional
        File contain sigmf meta.
    buffer : ndarray, optional
        Pinned memory buffer to use when copying data from GPU.
    num_samples : int, optional
        Number of samples to be loaded to GPU. If set to 0,
        read in all samples.
    offset : int, optional
        May be specified as a non-negative integer offset.
        It is the number of samples before loading 'num_samples'.
        'offset' must be a multiple of ALLOCATIONGRANULARITY which
        is equal to PAGESIZE on Unix systems.

    Returns
    -------
    out : ndarray
        An 1-dimensional array containing unpacked binary data.

    """

    if meta_file is None:
        meta_ext = ".sigmf-meta"

        pat = re.compile(r"(.+)(\.)(.+)")
        split_string = pat.split(data_file)
        meta_file = split_string[1] + meta_ext

    with open(meta_file, "r") as f:
        header = json.loads(f.read())

    dataset_type = _extract_values(header, "core:datatype")

    data_type = dataset_type[0].split("_")

    if len(data_type) == 1:
        endianness = "N"
    elif len(data_type) == 2:
        if data_type[1] == "le":
            endianness = "L"
        elif data_type[1] == "be":
            endianness = "B"
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    # Complex
    if data_type[0][0] == "c":
        if data_type[0][1:] == "f64":
            data_type = cp.complex128
        elif data_type[0][1:] == "f32":
            data_type = cp.complex64
        elif data_type[0][1:] == "i32":
            data_type = cp.int32
        elif data_type[0][1:] == "u32":
            data_type = cp.uint32
        elif data_type[0][1:] == "i16":
            data_type = cp.int16
        elif data_type[0][1:] == "u16":
            data_type = cp.uint16
        elif data_type[0][1:] == "i8":
            data_type = cp.int8
        elif data_type[0][1:] == "u8":
            data_type = cp.uint8
        else:
            raise NotImplementedError
    # Real
    elif data_type[0][0] == "r":
        if data_type[0][1:] == "f64":
            data_type = cp.float64
        elif data_type[0][1:] == "f32":
            data_type = cp.float32
        elif data_type[0][1:] == "i32":
            data_type = cp.int32
        elif data_type[0][1:] == "u32":
            data_type = cp.uint32
        elif data_type[0][1:] == "i16":
            data_type = cp.int16
        elif data_type[0][1:] == "u16":
            data_type = cp.uint16
        elif data_type[0][1:] == "i8":
            data_type = cp.int8
        elif data_type[0][1:] == "u8":
            data_type = cp.uint8
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError

    binary = read_bin(data_file, buffer, data_type, num_samples, offset)

    out = unpack_bin(binary, data_type, endianness)

    return out
