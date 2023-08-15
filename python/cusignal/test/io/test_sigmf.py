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

import cupy as cp

import cusignal


def test_read_paged(tmpdir):
    data_fname = tmpdir.join("test_read.sigmf-data")
    meta_fname = tmpdir.join("test_read.sigmf-meta")

    actual = cp.random.rand(100).astype(cp.complex64)
    meta = {"core:datatype": "cf32"}

    actual.tofile(data_fname)
    meta_fname.write(json.dumps(meta))

    expect = cusignal.read_sigmf(str(data_fname), str(meta_fname))

    cp.testing.assert_array_equal(actual, expect)


def test_read_pinned_buffer(tmpdir):
    data_fname = tmpdir.join("test_read.sigmf-data")
    meta_fname = tmpdir.join("test_read.sigmf-meta")

    actual = cp.random.rand(100).astype(cp.complex64)
    meta = {"core:datatype": "cf32"}

    actual.tofile(data_fname)
    meta_fname.write(json.dumps(meta))

    binary = cusignal.read_bin(str(data_fname))
    buffer = cusignal.get_pinned_mem(binary.shape, cp.ubyte)

    expect = cusignal.read_sigmf(str(data_fname), str(meta_fname), buffer)

    cp.testing.assert_array_equal(actual, expect)


def test_read_shared_buffer(tmpdir):
    data_fname = tmpdir.join("test_read.sigmf-data")
    meta_fname = tmpdir.join("test_read.sigmf-meta")

    actual = cp.random.rand(100).astype(cp.complex64)
    meta = {"core:datatype": "cf32"}

    actual.tofile(data_fname)
    meta_fname.write(json.dumps(meta))

    binary = cusignal.read_bin(str(data_fname))
    buffer = cusignal.get_shared_mem(binary.shape, cp.ubyte)

    expect = cusignal.read_sigmf(str(data_fname), str(meta_fname), buffer)

    cp.testing.assert_array_equal(actual, expect)


def test_write_paged(tmpdir):
    data_fname = tmpdir.join("test_read.sigmf-data")
    meta_fname = tmpdir.join("test_read.sigmf-meta")

    actual = cp.random.rand(100).astype(cp.complex64)
    meta = {"core:datatype": "cf32"}

    cusignal.write_sigmf(str(data_fname), actual)
    meta_fname.write(json.dumps(meta))

    expect = cusignal.read_sigmf(str(data_fname), str(meta_fname))

    cp.testing.assert_array_equal(actual, expect)


def test_write_pinned_buffer(tmpdir):
    data_fname = tmpdir.join("test_read.sigmf-data")
    meta_fname = tmpdir.join("test_read.sigmf-meta")

    actual = cp.random.rand(100).astype(cp.complex64)
    meta = {"core:datatype": "cf32"}

    cusignal.write_bin(str(data_fname), actual)
    meta_fname.write(json.dumps(meta))

    binary = cusignal.read_bin(str(data_fname))
    buffer = cusignal.get_pinned_mem(binary.shape, cp.ubyte)

    cusignal.write_sigmf(str(data_fname), actual, buffer=buffer, append=False)

    expect = cusignal.read_sigmf(str(data_fname), str(meta_fname))

    cp.testing.assert_array_equal(actual, expect)


def test_write_shared_buffer(tmpdir):
    data_fname = tmpdir.join("test_read.sigmf-data")
    meta_fname = tmpdir.join("test_read.sigmf-meta")

    actual = cp.random.rand(100).astype(cp.complex64)
    meta = {"core:datatype": "cf32"}

    cusignal.write_bin(str(data_fname), actual)
    meta_fname.write(json.dumps(meta))

    binary = cusignal.read_bin(str(data_fname))
    buffer = cusignal.get_shared_mem(binary.shape, cp.ubyte)

    cusignal.write_sigmf(str(data_fname), actual, buffer=buffer, append=False)

    expect = cusignal.read_sigmf(str(data_fname), str(meta_fname))

    cp.testing.assert_array_equal(actual, expect)
