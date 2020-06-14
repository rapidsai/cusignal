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


def read_bin(
    file, cp_stream=cp.cuda.stream.Stream.null
):
    """
    Reads binary file input GPU memory.

    Parameters
    ----------
    file : str
        A string of filename to be read to GPU.
    cp_stream : CuPy stream, optional
        Option allows upfirdn to run in a non-default stream. The use
        of multiple non-default streams allow multiple kernels to
        run concurrently. Default is cp.cuda.stream.Stream.null
        or default stream.

    Returns
    -------
    out : ndarray
        An 1-dimensional array containing parsed binary data.

    """

    with open(file, "r+") as f:
        mm = mmap(f.fileno(), 0, flags=MAP_PRIVATE, prot=PROT_READ,)

        with cp_stream:
            out = cp.asarray(mm)
        # Must synchronize stream before closing mmap
        cp_stream.synchronize()

        mm.close()

    return out


def parse_bin(
    in1,
    format,
    keep=True,
    dtype=np.complex64,
    cp_stream=cp.cuda.stream.Stream.null,
    autosync=True,
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
    cp_stream : CuPy stream, optional
        Option allows upfirdn to run in a non-default stream. The use
        of multiple non-default streams allow multiple kernels to
        run concurrently. Default is cp.cuda.stream.Stream.null
        or default stream.
    autosync : bool, optional
        Option to automatically synchronize cp_stream. This will block
        the host code until kernel is finished on the GPU. Setting to
        false will allow asynchronous operation but might required
        manual synchronize later `cp_stream.synchronize()`.
        Default is True.

    Returns
    -------
    out : ndarray
        An 1-dimensional array containing parsed binary data.

    """

    out = _parser(in1, format, keep, dtype, cp_stream, autosync)

    return out


def fromfile(
    file,
    format,
    keep=True,
    dtype=np.complex64,
    cp_stream=cp.cuda.stream.Stream.null,
    autosync=True,
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
    cp_stream : CuPy stream, optional
        Option allows upfirdn to run in a non-default stream. The use
        of multiple non-default streams allow multiple kernels to
        run concurrently. Default is cp.cuda.stream.Stream.null
        or default stream.
    autosync : bool, optional
        Option to automatically synchronize cp_stream. This will block
        the host code until kernel is finished on the GPU. Setting to
        false will allow asynchronous operation but might required
        manual synchronize later `cp_stream.synchronize()`.
        Default is True.

    Returns
    -------
    out : ndarray
        An 1-dimensional array containing parsed binary data.

    """

    binary = read_bin(file, cp_stream)
    out = parse_bin(
        binary,
        format="sigmf",
        keep=keep,
        dtype=dtype,
        cp_stream=cp_stream,
        autosync=autosync,
    )

    return out
