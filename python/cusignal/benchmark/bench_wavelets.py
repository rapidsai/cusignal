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

import pytest
import cupy as cp
import numpy as np
from cusignal.test.utils import array_equal
import cusignal
from scipy import signal

cusignal.precompile_kernels()


@pytest.mark.benchmark(group="Morlet")
@pytest.mark.parametrize("num_samps", [2 ** 14])
class BenchMorlet:
    def cpu_version(self, num_samps):
        return signal.morlet(num_samps)

    def bench_morlet_cpu(self, benchmark, num_samps):
        benchmark(self.cpu_version, num_samps)

    def bench_morlet_gpu(self, benchmark, num_samps):

        output = benchmark(cusignal.morlet, num_samps)

        key = self.cpu_version(num_samps)
        assert array_equal(cp.asnumpy(output), key)


@pytest.mark.benchmark(group="Ricker")
@pytest.mark.parametrize("num_samps", [2 ** 14])
@pytest.mark.parametrize("a", [10, 1000])
class BenchRicker:
    def cpu_version(self, num_samps, a):
        return signal.ricker(num_samps, a)

    def bench_ricker_cpu(self, benchmark, num_samps, a):
        benchmark(self.cpu_version, num_samps, a)

    def bench_ricker_gpu(self, benchmark, num_samps, a):

        output = benchmark(cusignal.ricker, num_samps, a)

        key = self.cpu_version(num_samps, a)
        assert array_equal(cp.asnumpy(output), key)


@pytest.mark.benchmark(group="CWT")
@pytest.mark.parametrize("num_samps", [2 ** 14])
@pytest.mark.parametrize("widths", [31, 127])
class BenchCWT:
    def cpu_version(self, cpu_sig, wavelet, widths):
        return signal.cwt(cpu_sig, wavelet, np.arange(1, widths))

    def bench_cwt_cpu(self, rand_data_gen, benchmark, num_samps, widths):
        cpu_sig, _ = rand_data_gen(num_samps)
        wavelet = signal.ricker
        benchmark(self.cpu_version, cpu_sig, wavelet, widths)

    def bench_cwt_gpu(self, rand_data_gen, benchmark, num_samps, widths):

        cpu_sig, gpu_sig = rand_data_gen(num_samps)
        cu_wavelet = cusignal.ricker
        output = benchmark(
            cusignal.cwt, gpu_sig, cu_wavelet, cp.arange(1, widths)
        )

        wavelet = signal.ricker
        key = self.cpu_version(cpu_sig, wavelet, widths)
        assert array_equal(cp.asnumpy(output), key)


@pytest.mark.benchmark(group="CWTComplex")
@pytest.mark.parametrize("num_samps", [2 ** 14])
@pytest.mark.parametrize("widths", [31, 127])
class BenchCWTComplex:
    def cpu_version(self, cpu_sig, wavelet, widths):
        return signal.cwt(cpu_sig, wavelet, np.arange(1, widths))

    def bench_cwt_complex_cpu(
        self, rand_complex_data_gen, benchmark, num_samps, widths
    ):
        cpu_sig, _ = rand_complex_data_gen(num_samps)
        wavelet = signal.ricker
        benchmark(self.cpu_version, cpu_sig, wavelet, widths)

    def bench_cwt_complex_gpu(
        self, rand_complex_data_gen, benchmark, num_samps, widths
    ):

        cpu_sig, gpu_sig = rand_complex_data_gen(num_samps)
        cu_wavelet = cusignal.ricker
        output = benchmark(
            cusignal.cwt, gpu_sig, cu_wavelet, cp.arange(1, widths)
        )

        wavelet = signal.ricker
        key = self.cpu_version(cpu_sig, wavelet, widths)
        assert array_equal(cp.asnumpy(output), key)
