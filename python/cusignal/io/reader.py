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

from mmap import mmap, MAP_PRIVATE, PROT_READ

from ._reader_cuda import _parser


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

    with open(file, "r+") as f:
        mm = mmap(f.fileno(), 0, flags=MAP_PRIVATE, prot=PROT_READ,)
        out = cp.asarray(mm)
        mm.close()

    return out


def parse_bin(
    in1, format, keep=True, dtype=np.complex64,
):
    """
    Parse binary file

    Parameters
    ----------
    in1 : array_like
        The binary array to be parsed.
    format : str
        Dataset format specification to be used when unpacking binary.
    keep : bool, optional
        Option whether to delete binary data after parsing..
    dtype : data-type, optional
        Any object that can be interpreted as a numpy data type.

    Returns
    -------
    out : ndarray
        An 1-dimensional array containing parsed binary data.

    """

    out = _parser(in1, format, keep, dtype)

    return out


def fromfile(
    file, format, keep=True, dtype=np.complex64,
):
    """
    Read and parse binary file to GPU memory

    Parameters
    ----------
    file : str
        A string of filename to be read/parsed/upacked to GPU.
    format : str
        Dataset format specification to be used when unpacking binary.
    keep : bool, optional
        Option whether to delete binary data on GPU after parsing.
    dtype : data-type, optional
        Any object that can be interpreted as a numpy data type.

    Returns
    -------
    out : ndarray
        An 1-dimensional array containing parsed binary data.

    """

    binary = read_bin(file)
    out = parse_bin(binary, format="sigmf", keep=keep, dtype=dtype,)

    return out
