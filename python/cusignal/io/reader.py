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

from mmap import mmap, MAP_PRIVATE, PROT_READ

from ._reader_cuda import _unpack


def read_bin(file):
    """
    Reads binary file input GPU memory.

    Parameters
    ----------
    file : str
        A string of filename to be read to GPU.

    Returns
    -------
    out : ndarray
        An 1-dimensional array containing parsed binary data.

    """

    # Get current stream, default or not.
    stream = cp.cuda.get_current_stream()

    with open(file, "rb") as f:
        mm = mmap(f.fileno(), 0, flags=MAP_PRIVATE, prot=PROT_READ,)
        out = cp.asarray(mm)
        stream.synchronize()
        mm.close()

    return out


def unpack_bin(in1, spec, dtype, endianness="L"):
    """
    Unpack binary file. If endianness is big-endian, it my be converted
    to little endian.

    Parameters
    ----------
    in1 : array_like
        The binary array to be unpack.
    spec : str
        Dataset specification to be used when unpacking binary.
    dtype : data-type, optional
        Any object that can be interpreted as a numpy data type.
    endianness : {'L', 'B'}, optional
        Data set byte order

    Returns
    -------
    out : ndarray
        An 1-dimensional array containing unpacked binary data.

    """

    out = _unpack(in1, spec, dtype, endianness)

    return out


def read_sigmf(file):
    """
    Read and parse binary file to GPU memory

    Parameters
    ----------
    file : str
        A string of filename to be read/parsed/upacked to GPU.
    # spec : str
    #     Dataset specification to be used when unpacking binary.
    keep : bool, optional
        Option whether to delete binary data on GPU after parsing.
    dtype : data-type, optional
        Any object that can be interpreted as a numpy data type.

    Returns
    -------
    out : ndarray
        An 1-dimensional array containing parsed binary data.

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
        print(data_type[1])
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

    binary = read_bin(file + data_ext)

    out = unpack_bin(
        binary, dtype=data_type, endianness=endianness
    )

    return out
