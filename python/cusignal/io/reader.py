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

import json
import resource

from mmap import mmap, MAP_PRIVATE, PROT_READ

from ._reader_cuda import _unpack


def read_bin(file, dtype=None, num_samples=0, offset=0):
    """
    Reads binary file input GPU memory.

    Parameters
    ----------
    file : str
        A string of filename to be read to GPU.
    dtype : data-type, optional
        Any object that can be interpreted as a numpy data type.
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
        An 1-dimensional array containing binary data.

    """

    # Get current stream, default or not.
    stream = cp.cuda.get_current_stream()

    if num_samples < 0 or offset < 0:
        raise ValueError("'num_samples' and 'offset' must be >= 0")

    if dtype is None and (num_samples != 0 or offset != 0):
        raise TypeError(
            "'dtype' must be provided is 'num_samples'/'offset' != 0"
        )

    num_bytes_read = num_samples * cp.dtype(dtype).itemsize
    offset *= cp.dtype(dtype).itemsize

    if (offset % resource.getpagesize()) != 0:
        raise ValueError("'offset' must be a multiple of the PAGESIZE")

    with open(file, "rb") as f:
        mm = mmap(
            f.fileno(),
            num_bytes_read,
            flags=MAP_PRIVATE,
            prot=PROT_READ,
            offset=offset,
        )
        out = cp.asarray(mm)
        stream.synchronize()
        mm.close()

    return out


def unpack_bin(in1, dtype, endianness="L"):
    """
    Unpack binary file. If endianness is big-endian, it my be converted
    to little endian for NVIDIA GPU compatibility.

    Parameters
    ----------
    in1 : array_like
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

    out = _unpack(in1, dtype, endianness)

    return out


def read_sigmf(file, num_samples=0, offset=0):
    """
    Read and unpack binary file to GPU memory

    Parameters
    ----------
    file : str
        A string of filename to be read/unpacked to GPU.
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

    meta_ext = ".sigmf-meta"
    data_ext = ".sigmf-data"

    with open(file + meta_ext, "r") as f:
        header = json.loads(f.read())

    dataset_type = header["_metadata"]["global"]["core:datatype"]

    data_type = dataset_type.split("_")

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
        if data_type[0][1:] == "f32":
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
        if data_type[0][1:] == "f32":
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

    binary = read_bin(file + data_ext, data_type, num_samples, offset)

    out = unpack_bin(binary, data_type, endianness)

    return out
