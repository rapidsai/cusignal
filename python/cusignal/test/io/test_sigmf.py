# Copyright (c) 2022, NVIDIA CORPORATION.
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
