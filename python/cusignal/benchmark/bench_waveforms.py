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
import pytest-benchmark
import cupy as cp
from cusignal.test.utils import array_equal
import cusignal
from scipy import signal


@pytest.mark.benchmark(group="Square")
@pytest.mark.parametrize("num_samps", [2 ** 14])
@pytest.mark.parametrize("duty", [0.25, 0.5])
class BenchSquare:
    def cpu_version(self, cpu_sig, duty):
        return signal.square(cpu_sig, duty)

    def bench_square_cpu(self, time_data_gen, benchmark, num_samps, duty):
        cpu_sig, _ = time_data_gen(0, 10, num_samps)
        benchmark(self.cpu_version, cpu_sig, duty)

    def bench_square_gpu(self, time_data_gen, benchmark, num_samps, duty):

        cpu_sig, gpu_sig = time_data_gen(0, 10, num_samps)
        output = benchmark(cusignal.square, gpu_sig, duty)

        key = self.cpu_version(cpu_sig, duty)
        assert array_equal(cp.asnumpy(output), key)


@pytest.mark.benchmark(group="GaussPulse")
@pytest.mark.parametrize("num_samps", [2 ** 14])
@pytest.mark.parametrize("fc", [0.75, 5])
class BenchGaussPulse:
    def cpu_version(self, cpu_sig, fc):
        return signal.gausspulse(cpu_sig, fc, retquad=True, retenv=True)

    def bench_gausspulse_cpu(self, time_data_gen, benchmark, num_samps, fc):
        cpu_sig, _ = time_data_gen(0, 10, num_samps)
        benchmark(self.cpu_version, cpu_sig, fc)

    def bench_gausspulse_gpu(self, time_data_gen, benchmark, num_samps, fc):

        cpu_sig, gpu_sig = time_data_gen(0, 10, num_samps)
        output = benchmark(
            cusignal.gausspulse, gpu_sig, fc, retquad=True, retenv=True
        )

        key = self.cpu_version(cpu_sig, fc)
        assert array_equal(cp.asnumpy(output), key)


@pytest.mark.benchmark(group="Chirp")
@pytest.mark.parametrize("num_samps", [2 ** 14])
@pytest.mark.parametrize("f0", [6])
@pytest.mark.parametrize("t1", [1])
@pytest.mark.parametrize("f1", [10])
@pytest.mark.parametrize("method", ["linear", "quadratic"])
class BenchChirp:
    def cpu_version(self, cpu_sig, f0, t1, f1, method):
        return signal.chirp(cpu_sig, f0, t1, f1, method)

    def bench_chirp_cpu(
        self, time_data_gen, benchmark, num_samps, f0, t1, f1, method
    ):
        cpu_sig, _ = time_data_gen(0, 10, num_samps)
        benchmark(self.cpu_version, cpu_sig, f0, t1, f1, method)

    def bench_chirp_gpu(
        self, time_data_gen, benchmark, num_samps, f0, t1, f1, method
    ):

        cpu_sig, gpu_sig = time_data_gen(0, 10, num_samps)
        output = benchmark(cusignal.chirp, gpu_sig, f0, t1, f1, method)

        key = self.cpu_version(cpu_sig, f0, t1, f1, method)
        assert array_equal(cp.asnumpy(output), key)


@pytest.mark.benchmark(group="UnitImpulse")
@pytest.mark.parametrize("num_samps", [2 ** 14])
@pytest.mark.parametrize("idx", ["mid"])
class BenchUnitImpulse:
    def cpu_version(self, num_samps, idx):
        return signal.unit_impulse(num_samps, idx)

    def bench_unit_impulse_cpu(self, benchmark, num_samps, idx):
        benchmark(self.cpu_version, num_samps, idx)

    def bench_unit_impulse_gpu(self, benchmark, num_samps, idx):

        output = benchmark(cusignal.unit_impulse, num_samps, idx)

        key = self.cpu_version(num_samps, idx)
        assert array_equal(cp.asnumpy(output), key)
