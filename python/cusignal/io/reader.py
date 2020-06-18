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
import numpy as np

import json
import resource

from mmap import mmap, MAP_PRIVATE, PROT_READ

from ._reader_cuda import _unpack, _pack


# https://github.com/cupy/cupy/blob/master/examples/stream/cupy_memcpy.py
def _pin_memory(array):
    mem = cp.cuda.alloc_pinned_memory(array.nbytes)
    ret = np.frombuffer(mem, array.dtype, array.size).reshape(array.shape)
    ret[...] = array
    return ret


def pin_memory(size, dtype):
    """
    Create a pinned memory buffer.

    Parameters
    ----------
    size : int or tuple of ints
        Output shape.
    dtype : data-type
        Output data type.

    Returns
    -------
    out : ndarray
        Pinned memory numpy array.

    """
    pinned_memory_pool = cp.cuda.PinnedMemoryPool()
    cp.cuda.set_pinned_memory_allocator(pinned_memory_pool.malloc)

    x_cpu_dst = np.empty(size, dtype)
    x_pinned_cpu_dst = _pin_memory(x_cpu_dst)

    return x_pinned_cpu_dst


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


def write_bin(file, binary, buffer=None, append=True):
    """
    Writes binary array to file.

    Parameters
    ----------
    file : str
        A string of filename to store output.
    binary : ndarray
        Binary array to be written to file.
    buffer : ndarray, optional
        Pinned memory buffer to use when copying data from GPU.
    append : bool, optional
        Append to file if created.

    Returns
    -------
    out : ndarray
        An 1-dimensional array containing binary data.

    """

    # Get current stream, default or not.
    stream = cp.cuda.get_current_stream()

    if buffer is None:
        buffer = cp.asnumpy(binary)
    else:
        binary.get(out=buffer)

    if append is True:
        mode = "ab"
    else:
        mode = "wb"

    with open(file, mode) as f:
        stream.synchronize()
        f.write(buffer)


def unpack_bin(binary, dtype, endianness="L"):
    """
    Unpack binary file. If endianness is big-endian, it my be converted
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


def pack_bin(in1):
    """
    Pack binary arrary.
    Data will be packed with little endian for NVIDIA GPU compatibility.

    Parameters
    ----------
    in1 : ndarray
        The ndarray to be pack at binary.

    Returns
    -------
    out : ndarray
        An 1-dimensional array containing packed binary data.

    """

    out = _pack(in1)

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
        if data_type[0][1:] == "f64":
            data_type = cp.complex128
        elif data_type[0][1:] == "f32":
            data_type = cp.complex128
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

    binary = read_bin(file + data_ext, data_type, num_samples, offset)

    out = unpack_bin(binary, data_type, endianness)

    return out


def write_sigmf(file, data, buffer=None, append=True):
    """
    Pack and write binary array to file>

    Parameters
    ----------
    file : str
        A string of filename to be read/unpacked to GPU.
    binary : ndarray
        Binary array to be written to file.
    buffer : ndarray, optional
        Pinned memory buffer to use when copying data from GPU.
    append : bool, optional
        Append to file if created.

    Returns
    -------

    """

    data_ext = ".sigmf-data"

    packed = pack_bin(data)

    write_bin(file + data_ext, packed, buffer, append)
