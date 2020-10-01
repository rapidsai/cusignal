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
import cusignal
import numpy as np
import pytest

from cusignal.test.utils import array_equal, _check_rapids_pytest_benchmark
from scipy import signal

gpubenchmark = _check_rapids_pytest_benchmark()

# # Missing
# # cubic
# # quadratic
# # cspline1d


class TestBsplines:
    @pytest.mark.parametrize("x", [-1.0, 0.0, -1.0])
    @pytest.mark.parametrize("n", [1])
    @pytest.mark.benchmark(group="GaussSpline")
    class TestGaussSpline:
        def cpu_version(self, x, n):
            return signal.gauss_spline(x, n)

        def gpu_version(self, x, n):
            with cp.cuda.Stream.null:
                out = cusignal.gauss_spline(x, n)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_gauss_spline_cpu(self, benchmark, x, n):
            benchmark(self.cpu_version, x, n)

        def test_gauss_spline_gpu(self, gpubenchmark, x, n):

            d_x = cp.asarray(x)
            output = gpubenchmark(self.gpu_version, d_x, n)

            key = self.cpu_version(x, n)
            assert array_equal(cp.asnumpy(output), key)


    @pytest.mark.parametrize("x", [2 ** 16])
    @pytest.mark.benchmark(group="Cubic")
    class TestCubic:
        def cpu_version(self, x):
            return signal.cubic(x)

        def gpu_version(self, d_x):
            with cp.cuda.Stream.null:
                out = cusignal.cubic(d_x)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_cubic_cpu(self, benchmark, rand_data_gen, x):

            cpu_sig, _ = rand_data_gen(x)
            benchmark(self.cpu_version, cpu_sig)

        def test_cubic_gpu(self, gpubenchmark, rand_data_gen, x):

            cpu_sig, gpu_sig = rand_data_gen(x)
            output = gpubenchmark(self.gpu_version, gpu_sig)

            key = self.cpu_version(cpu_sig)
            assert array_equal(cp.asnumpy(output), key)

    @pytest.mark.parametrize("x", [-1.0, 0.0, -1.0])
    @pytest.mark.benchmark(group="Quadratic")
    class TestQuadratic:
        def cpu_version(self, x):
            return signal.quadratic(x)

        def gpu_version(self, d_x):
            with cp.cuda.Stream.null:
                out = cusignal.quadratic(d_x)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_quadratic_cpu(self, benchmark, x):
            benchmark(self.cpu_version, x)

        def test_quadratic_gpu(self, gpubenchmark, x):

            d_x = cp.asarray(x)
            output = gpubenchmark(self.gpu_version, d_x)

            key = self.cpu_version(x)
            assert array_equal(cp.asnumpy(output), key)

#     @pytest.mark.benchmark(group="Cspline1d")
#     class TestCspline1d:
#         def cpu_version(self, cpu_sig):
#             return signal.cspline1d(cpu_sig)

#         @pytest.mark.cpu
#         def test_cspline1d_cpu(self, benchmark):
#             benchmark(self.cpu_version, cpu_sig)

#         def test_cspline1d_gpu(self, gpubenchmark):

#             output = gpubenchmark(cusignal.cspline1d, gpu_sig)

#             key = self.cpu_version(cpu_sig)
#             assert array_equal(cp.asnumpy(output), key)
