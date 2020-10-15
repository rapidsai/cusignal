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
import pytest

from cusignal.test.utils import array_equal, _check_rapids_pytest_benchmark
from scipy import signal

gpubenchmark = _check_rapids_pytest_benchmark()


class TestFilterDesign:
    @pytest.mark.benchmark(group="FirWin")
    @pytest.mark.parametrize("num_samps", [2 ** 15])
    @pytest.mark.parametrize("f1", [0.1, 0.15])
    @pytest.mark.parametrize("f2", [0.2, 0.4])
    class TestFirWin:
        def cpu_version(self, num_samps, f1, f2):
            return signal.firwin(num_samps, [f1, f2], pass_zero=False)

        def gpu_version(self, num_samps, f1, f2):
            with cp.cuda.Stream.null:
                out = cusignal.firwin(num_samps, [f1, f2], pass_zero=False)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_firwin_cpu(self, benchmark, num_samps, f1, f2):
            benchmark(
                self.cpu_version,
                num_samps,
                f1,
                f2,
            )

        def test_firwin_gpu(self, gpubenchmark, num_samps, f1, f2):

            output = gpubenchmark(
                self.gpu_version,
                num_samps,
                f1,
                f2,
            )

            key = self.cpu_version(num_samps, f1, f2)
            array_equal(cp.asnumpy(output), key)

    # Not passing anything to cupy, faster in numba
    @pytest.mark.parametrize("a", [5, 25, 100])
    @pytest.mark.benchmark(group="KaiserBeta")
    class TestKaiserBeta:
        def cpu_version(self, a):
            return signal.kaiser_beta(a)

        def gpu_version(self, a):
            with cp.cuda.Stream.null:
                out = cusignal.kaiser_beta(a)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_kaiser_beta_cpu(self, benchmark, a):
            benchmark(self.cpu_version, a)

        def test_kaiser_beta_gpu(self, gpubenchmark, a):

            output = gpubenchmark(self.gpu_version, a)

            key = self.cpu_version(a)
            array_equal(cp.asnumpy(output), key)

    # Not passing anything to cupy, faster in numba
    @pytest.mark.parametrize("numtaps", [5, 25, 100])
    @pytest.mark.parametrize("width", [0.01, 0.0375, 2.4])
    @pytest.mark.benchmark(group="KaiserAtten")
    class TestKaiserAtten:
        def cpu_version(self, numtaps, width):
            return signal.kaiser_atten(numtaps, width)

        def gpu_version(self, numtaps, width):
            with cp.cuda.Stream.null:
                out = cusignal.kaiser_atten(numtaps, width)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_kaiser_atten_cpu(self, benchmark, numtaps, width):
            benchmark(self.cpu_version, numtaps, width)

        def test_kaiser_atten_gpu(self, gpubenchmark, numtaps, width):

            output = gpubenchmark(self.gpu_version, numtaps, width)

            key = self.cpu_version(numtaps, width)
            array_equal(cp.asnumpy(output), key)

    # Not passing anything to cupy, faster in numba
    @pytest.mark.benchmark(group="CmplxSort")
    @pytest.mark.parametrize("p", [1 + 2j, 2 - 1j, 3 - 2j, 3 - 3j, 3 + 5j])
    class TestCmplxSort:
        def cpu_version(self, p):
            return signal.cmplx_sort(p)

        def gpu_version(self, p):
            with cp.cuda.Stream.null:
                out = cusignal.cmplx_sort(p)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_cmplx_sort_cpu(self, benchmark, p):
            benchmark(self.cpu_version, p)

        def test_cmplx_sort_gpu(self, gpubenchmark, p):

            output, _ = gpubenchmark(self.gpu_version, p)

            key, _ = self.cpu_version(p)
            array_equal(cp.asnumpy(output), key)
