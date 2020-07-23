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
import pytest_benchmark

from cusignal.test.utils import array_equal
from scipy import signal

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


class TestWaveforms:
    @pytest.mark.benchmark(group="Square")
    @pytest.mark.parametrize("num_samps", [2 ** 14])
    @pytest.mark.parametrize("duty", [0.25, 0.5])
    class TestSquare:
        def cpu_version(self, cpu_sig, duty):
            return signal.square(cpu_sig, duty)

        @pytest.mark.cpu
        def test_square_cpu(self, time_data_gen, benchmark, num_samps, duty):
            cpu_sig, _ = time_data_gen(0, 10, num_samps)
            benchmark(self.cpu_version, cpu_sig, duty)

        def test_square_gpu(
            self, time_data_gen, gpubenchmark, num_samps, duty
        ):

            cpu_sig, gpu_sig = time_data_gen(0, 10, num_samps)
            output = gpubenchmark(cusignal.square, gpu_sig, duty)

            key = self.cpu_version(cpu_sig, duty)
            assert array_equal(cp.asnumpy(output), key)

    @pytest.mark.benchmark(group="GaussPulse")
    @pytest.mark.parametrize("num_samps", [2 ** 14])
    @pytest.mark.parametrize("fc", [0.75, 5])
    class TestGaussPulse:
        def cpu_version(self, cpu_sig, fc):
            return signal.gausspulse(cpu_sig, fc, retquad=True, retenv=True)

        @pytest.mark.cpu
        def test_gausspulse_cpu(self, time_data_gen, benchmark, num_samps, fc):
            cpu_sig, _ = time_data_gen(0, 10, num_samps)
            benchmark(self.cpu_version, cpu_sig, fc)

        def test_gausspulse_gpu(
            self, time_data_gen, gpubenchmark, num_samps, fc
        ):

            cpu_sig, gpu_sig = time_data_gen(0, 10, num_samps)
            _, _, output = gpubenchmark(
                cusignal.gausspulse, gpu_sig, fc, retquad=True, retenv=True
            )

            _, _, key = self.cpu_version(cpu_sig, fc)
            assert array_equal(cp.asnumpy(output), key)

    @pytest.mark.benchmark(group="Chirp")
    @pytest.mark.parametrize("num_samps", [2 ** 14])
    @pytest.mark.parametrize("f0", [6])
    @pytest.mark.parametrize("t1", [1])
    @pytest.mark.parametrize("f1", [10])
    @pytest.mark.parametrize("method", ["linear", "quadratic"])
    class TestChirp:
        def cpu_version(self, cpu_sig, f0, t1, f1, method):
            return signal.chirp(cpu_sig, f0, t1, f1, method)

        @pytest.mark.cpu
        def test_chirp_cpu(
            self, time_data_gen, benchmark, num_samps, f0, t1, f1, method
        ):
            cpu_sig, _ = time_data_gen(0, 10, num_samps)
            benchmark(self.cpu_version, cpu_sig, f0, t1, f1, method)

        def test_chirp_gpu(
            self, time_data_gen, gpubenchmark, num_samps, f0, t1, f1, method
        ):

            cpu_sig, gpu_sig = time_data_gen(0, 10, num_samps)
            output = gpubenchmark(cusignal.chirp, gpu_sig, f0, t1, f1, method)

            key = self.cpu_version(cpu_sig, f0, t1, f1, method)
            assert array_equal(cp.asnumpy(output), key)

    @pytest.mark.benchmark(group="UnitImpulse")
    @pytest.mark.parametrize("num_samps", [2 ** 14])
    @pytest.mark.parametrize("idx", ["mid"])
    class TestUnitImpulse:
        def cpu_version(self, num_samps, idx):
            return signal.unit_impulse(num_samps, idx)

        @pytest.mark.cpu
        def test_unit_impulse_cpu(self, benchmark, num_samps, idx):
            benchmark(self.cpu_version, num_samps, idx)

        def test_unit_impulse_gpu(self, gpubenchmark, num_samps, idx):

            output = gpubenchmark(cusignal.unit_impulse, num_samps, idx)

            key = self.cpu_version(num_samps, idx)
            assert array_equal(cp.asnumpy(output), key)
