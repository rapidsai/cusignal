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
