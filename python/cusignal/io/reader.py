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

    # prioritize dtype of buffer if provided
    if buffer is not None:
        dtype = buffer.dtype

    # offset is measured in bytes
    offset *= cp.dtype(dtype).itemsize

    fp = np.memmap(file, mode="r", offset=offset, shape=num_samples, dtype=dtype)

    if buffer is not None:
        buffer[:] = fp[:]
        out = cp.empty(buffer.shape, buffer.dtype)
        out.set(buffer)
    else:
        out = cp.asarray(fp)

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
