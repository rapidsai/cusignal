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


class TestWavelets:
    @pytest.mark.benchmark(group="Qmf")
    @pytest.mark.parametrize("num_samps", [2 ** 14])
    class TestQmf:
        def cpu_version(self, sig):
            return signal.qmf(sig)

        def gpu_version(self, sig):
            with cp.cuda.Stream.null:
                out = cusignal.qmf(sig)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_qmf_cpu(self, range_data_gen, benchmark, num_samps):
            cpu_sig, _ = range_data_gen(num_samps)
            benchmark(self.cpu_version, cpu_sig)

        def test_qmf_gpu(self, range_data_gen, gpubenchmark, num_samps):

            cpu_sig, gpu_sig = range_data_gen(num_samps)
            output = gpubenchmark(self.gpu_version, gpu_sig)

            key = self.cpu_version(cpu_sig)
            assert array_equal(cp.asnumpy(output), key)

    @pytest.mark.benchmark(group="Morlet")
    @pytest.mark.parametrize("num_samps", [2 ** 14])
    class TestMorlet:
        def cpu_version(self, num_samps):
            return signal.morlet(num_samps)

        def gpu_version(self, num_samps):
            with cp.cuda.Stream.null:
                out = cusignal.morlet(num_samps)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_morlet_cpu(self, benchmark, num_samps):
            benchmark(self.cpu_version, num_samps)

        def test_morlet_gpu(self, gpubenchmark, num_samps):

            output = gpubenchmark(self.gpu_version, num_samps)

            key = self.cpu_version(num_samps)
            assert array_equal(cp.asnumpy(output), key)

    @pytest.mark.benchmark(group="Ricker")
    @pytest.mark.parametrize("num_samps", [2 ** 14])
    @pytest.mark.parametrize("a", [10, 1000])
    class TestRicker:
        def cpu_version(self, num_samps, a):
            return signal.ricker(num_samps, a)

        def gpu_version(self, num_samps, a):
            with cp.cuda.Stream.null:
                out = cusignal.ricker(num_samps, a)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_ricker_cpu(self, benchmark, num_samps, a):
            benchmark(self.cpu_version, num_samps, a)

        def test_ricker_gpu(self, gpubenchmark, num_samps, a):

            output = gpubenchmark(self.gpu_version, num_samps, a)

            key = self.cpu_version(num_samps, a)
            assert array_equal(cp.asnumpy(output), key)

    @pytest.mark.benchmark(group="CWT")
    @pytest.mark.parametrize("num_samps", [2 ** 14])
    @pytest.mark.parametrize("widths", [31, 127])
    @pytest.mark.parametrize("dim", [1])
    class TestCWT:
        def cpu_version(self, sig, wavelet, widths):
            return signal.cwt(sig, wavelet, np.arange(1, widths))

        def gpu_version(self, sig, wavelet, widths):
            with cp.cuda.Stream.null:
                out = cusignal.cwt(sig, wavelet, np.arange(1, widths))
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_cwt_cpu(
            self, rand_data_gen, benchmark, num_samps, dim, widths
        ):
            cpu_sig, _ = rand_data_gen(num_samps, dim)
            wavelet = signal.ricker
            benchmark(self.cpu_version, cpu_sig, wavelet, widths)

        def test_cwt_gpu(
            self, rand_data_gen, gpubenchmark, num_samps, dim, widths
        ):

            cpu_sig, gpu_sig = rand_data_gen(num_samps, dim)
            cu_wavelet = cusignal.ricker
            output = gpubenchmark(
                self.gpu_version, gpu_sig, cu_wavelet, widths
            )

            wavelet = signal.ricker
            key = self.cpu_version(cpu_sig, wavelet, widths)
            assert array_equal(cp.asnumpy(output), key)

    @pytest.mark.benchmark(group="CWTComplex")
    @pytest.mark.parametrize("num_samps", [2 ** 14])
    @pytest.mark.parametrize("widths", [31, 127])
    @pytest.mark.parametrize("dim", [1])
    class TestCWTComplex:
        def cpu_version(self, sig, wavelet, widths):
            return signal.cwt(sig, wavelet, np.arange(1, widths))

        def gpu_version(self, sig, wavelet, widths):
            with cp.cuda.Stream.null:
                out = cusignal.cwt(sig, wavelet, np.arange(1, widths))
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_cwt_complex_cpu(
            self, rand_complex_data_gen, benchmark, num_samps, dim, widths
        ):
            cpu_sig, _ = rand_complex_data_gen(num_samps, dim)
            wavelet = signal.ricker
            benchmark(self.cpu_version, cpu_sig, wavelet, widths)

        def test_cwt_complex_gpu(
            self, rand_complex_data_gen, gpubenchmark, num_samps, dim, widths
        ):

            cpu_sig, gpu_sig = rand_complex_data_gen(num_samps, dim)
            cu_wavelet = cusignal.ricker
            output = gpubenchmark(
                self.gpu_version, gpu_sig, cu_wavelet, widths
            )

            wavelet = signal.ricker
            key = self.cpu_version(cpu_sig, wavelet, widths)
            assert array_equal(cp.asnumpy(output), key)
