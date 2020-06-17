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

from cusignal.test.utils import array_equal
from scipy import signal

# Missing
# gauss_spline
# cubic
# quadratic
# cspline1d


class BenchBsplines:
    @pytest.mark.benchmark(group="GaussSpline")
    class BenchGaussSpline:
        def cpu_version(self, cpu_sig):
            return signal.gauss_spline(cpu_sig)

        def bench_gauss_spline_cpu(self, benchmark):
            benchmark(self.cpu_version, cpu_sig)

        def bench_gauss_spline_gpu(self, benchmark):

            output = benchmark(cusignal.gauss_spline, gpu_sig)

            key = self.cpu_version(cpu_sig)
            assert array_equal(cp.asnumpy(output), key)

    @pytest.mark.benchmark(group="Cubic")
    class BenchCubic:
        def cpu_version(self, cpu_sig):
            return signal.cubic(cpu_sig)

        def bench_cubic_cpu(self, benchmark):
            benchmark(self.cpu_version, cpu_sig)

        def bench_cubic_gpu(self, benchmark):

            output = benchmark(cusignal.cubic, gpu_sig)

            key = self.cpu_version(cpu_sig)
            assert array_equal(cp.asnumpy(output), key)

    @pytest.mark.benchmark(group="Quadratic")
    class BenchQuadratic:
        def cpu_version(self, cpu_sig):
            return signal.quadratic(cpu_sig)

        def bench_quadratic_cpu(self, benchmark):
            benchmark(self.cpu_version, cpu_sig)

        def bench_quadratic_gpu(self, benchmark):

            output = benchmark(cusignal.quadratic, gpu_sig)

            key = self.cpu_version(cpu_sig)
            assert array_equal(cp.asnumpy(output), key)

    @pytest.mark.benchmark(group="Cspline1d")
    class BenchCspline1d:
        def cpu_version(self, cpu_sig):
            return signal.cspline1d(cpu_sig)

        def bench_cspline1d_cpu(self, benchmark):
            benchmark(self.cpu_version, cpu_sig)

        def bench_cspline1d_gpu(self, benchmark):

            output = benchmark(cusignal.cspline1d, gpu_sig)

            key = self.cpu_version(cpu_sig)
            assert array_equal(cp.asnumpy(output), key)
