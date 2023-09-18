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

import cupy as cp

from ._writer_cuda import _pack


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
        buffer.tofile(f)


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


def write_sigmf(data_file, data, buffer=None, append=True):
    """
    Pack and write binary array to file, with SigMF spec.

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

    packed = pack_bin(data)

    write_bin(data_file, packed, buffer, append)
