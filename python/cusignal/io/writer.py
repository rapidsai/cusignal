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
