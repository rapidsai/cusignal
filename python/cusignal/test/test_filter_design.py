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


# Missing
# kaiser_beta
# kaiser_atten
# cmplx_sort


class TestFilterDesign:
    @pytest.mark.benchmark(group="FirWin")
    @pytest.mark.parametrize("num_samps", [2 ** 15])
    @pytest.mark.parametrize("f1", [0.1, 0.15])
    @pytest.mark.parametrize("f2", [0.2, 0.4])
    class TestFirWin:
        def cpu_version(self, num_samps, f1, f2):
            return signal.firwin(num_samps, [f1, f2], pass_zero=False)

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
                cusignal.firwin, num_samps, [f1, f2], pass_zero=False
            )

            key = self.cpu_version(num_samps, f1, f2)
            assert array_equal(cp.asnumpy(output), key)

    @pytest.mark.benchmark(group="KaiserBeta")
    @pytest.mark.parametrize("a", [1, 22, 51])
    class TestKaiserBeta:
        def cpu_version(self, a):
            return signal.kaiser_beta(a)

        def gpu_version(self, a):
            return cusignal.kaiser_beta(a)

        cp.cuda.Stream.null.synchronize()

        @pytest.mark.cpu
        def test_kaiser_beta_cpu(self, benchmark, a):
            benchmark(self.cpu_version, a)

        def test_kaiser_beta_gpu(self, gpubenchmark, a):

            output = gpubenchmark(self.gpu_version, a)

            key = self.cpu_version(a)
            assert array_equal(cp.asnumpy(output), key)

    # num_taps is int for FIR filter
    # width is float for width of transition region
    @pytest.mark.benchmark(group="KaiserAtten")
    @pytest.mark.parametrize("num_taps", [211])
    @pytest.mark.parametrize("width", [0.0375])
    class TestKaiserAtten:
        def cpu_version(self, num_taps, width):
            return signal.kaiser_atten(num_taps, width)

        def gpu_version(self, num_taps, width):
            return cusignal.kaiser_atten(num_taps, width)

        cp.cuda.Stream.null.synchronize()

        @pytest.mark.cpu
        def test_kaiser_atten_cpu(self, benchmark, num_taps, width):
            benchmark(self.cpu_version, num_taps, width)

        def test_kaiser_atten_gpu(self, gpubenchmark, num_taps, width):

            output = gpubenchmark(self.gpu_version, num_taps, width)

            key = self.cpu_version(num_taps, width)
            assert array_equal(cp.asnumpy(output), key)

    # # vals is array like
    # @pytest.mark.benchmark(group="CmplxSort")
    # @pytest.mark.benchmark("num_samps", [2 ** 15])
    # #@pytest.mark.parametrize("vals", [1, 4, 1+1.j, 3])
    # class TestCmplxSort:
    #     def cpu_version(self, vals):
    #         return signal.cmplx_sort(vals)

    #     @pytest.mark.cpu
    #     def test_cmplx_sort_cpu(self, rand_data_gen, benchmark, num_samps):
    #         vals, _ = rand_data_gen(num_samps)
    #         benchmark(self.cpu_version, vals)

    #     def test_cmplx_sort_gpu(self, rand_data_gen, gpubenchmark, num_samps):

    #         #d_vals = cp.asarray(vals)
    #         vals, d_vals = rand_data_gen(num_samps)
    #         output = gpubenchmark(cusignal.cmplx_sort, d_vals)

    #         key = self.cpu_version(vals)
    #         assert array_equal(cp.asnumpy(output), key)
