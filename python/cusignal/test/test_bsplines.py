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
import pytest
from scipy import signal

import cusignal
from cusignal.testing.utils import _check_rapids_pytest_benchmark, array_equal

gpubenchmark = _check_rapids_pytest_benchmark()


class TestBsplines:
    @pytest.mark.parametrize("x", [2**16])
    @pytest.mark.parametrize("n", [1])
    @pytest.mark.benchmark(group="GaussSpline")
    class TestGaussSpline:
        def cpu_version(self, x, n):
            return signal.gauss_spline(x, n)

        def gpu_version(self, d_x, n):
            with cp.cuda.Stream.null:
                out = cusignal.gauss_spline(d_x, n)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_gauss_spline_cpu(self, benchmark, rand_data_gen, x, n):

            cpu_sig, _ = rand_data_gen(x)
            benchmark(self.cpu_version, cpu_sig, n)

        def test_gauss_spline_gpu(self, gpubenchmark, rand_data_gen, x, n):

            cpu_sig, gpu_sig = rand_data_gen(x)
            output = gpubenchmark(self.gpu_version, gpu_sig, n)

            key = self.cpu_version(cpu_sig, n)
            array_equal(output, key)

    @pytest.mark.parametrize("x", [2**16])
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
            array_equal(output, key)

    @pytest.mark.parametrize("x", [2**16])
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
        def test_quadratic_cpu(self, benchmark, rand_data_gen, x):
            cpu_sig, _ = rand_data_gen(x)
            benchmark(self.cpu_version, cpu_sig)

        def test_quadratic_gpu(self, gpubenchmark, rand_data_gen, x):

            cpu_sig, gpu_sig = rand_data_gen(x)
            output = gpubenchmark(self.gpu_version, gpu_sig)

            key = self.cpu_version(cpu_sig)
            array_equal(output, key)
