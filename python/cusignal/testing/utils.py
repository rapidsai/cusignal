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
import pytest_benchmark


def array_equal(a, b, rtol=1e-7, atol=1e-5):

    if isinstance(a, tuple):  # Test functions with multiple outputs
        if a[0].dtype == cp.float32 or a[0].dtype == cp.complex64:
            rtol = 1e-3
            atol = 1e-3

        # Relaxed tolerances for single-precision arrays.
        if a[0].dtype == cp.float32 or b[0].dtype == cp.float32:
            rtol = 1e-1
            atol = 1e-1

        for i in range(len(a)):
            cp.testing.assert_allclose(a[i], b[i], rtol=rtol, atol=atol)

    elif not isinstance(a, (float, int)):
        if a.dtype == cp.float32 or a.dtype == cp.complex64:
            rtol = 1e-3
            atol = 1e-3

        # Relaxed tolerances for single-precision arrays.
        if a.dtype == cp.float32 or b.dtype == cp.float32:
            rtol = 1e-1
            atol = 1e-1

        cp.testing.assert_allclose(a, b, rtol=rtol, atol=atol)


def _check_rapids_pytest_benchmark():
    try:
        from rapids_pytest_benchmark import setFixtureParamNames
    except ImportError:
        print(
            "\n\nWARNING: rapids_pytest_benchmark is not installed, "
            "falling back to pytest_benchmark fixtures.\n"
        )

        # if rapids_pytest_benchmark is not available, just perfrom time-only
        # benchmarking and replace the util functions with nops
        gpubenchmark = pytest_benchmark.plugin.benchmark

        def setFixtureParamNames(*args, **kwargs):
            pass

        return gpubenchmark
