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

class TestPeakFinding:
    @pytest.mark.benchmark(group="Argrelmin")
    @pytest.mark.parametrize("num_samps", [2 ** 5])
    class TestArgrelmin:
        def cpu_version(self, sig):
            return signal.argrelmin(sig, mode="warp")

        def gpu_version(self, sig):
            with cp.cuda.Stream.null:
                out = cusignal.argrelmin(sig)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_argrelmin_cpu(self, rand_2d_data_gen, benchmark, num_samps):
            cpu_sig, _ = rand_2d_data_gen(num_samps)
            benchmark(self.cpu_version, cpu_sig)

        def test_argrelmin_gpu(
            self, rand_2d_data_gen, gpubenchmark, num_samps
        ):

            cpu_sig, gpu_sig = rand_2d_data_gen(num_samps)
            output = gpubenchmark(self.gpu_version, gpu_sig)

            key = self.cpu_version(cpu_sig)
            assert array_equal(cp.asnumpy(output), key)

    @pytest.mark.benchmark(group="Argrelmax")
    @pytest.mark.parametrize("num_samps", [2 ** 5])
    class TestArgrelmax:
        def cpu_version(self, sig):
            return signal.argrelmax(sig, mode="warp")

        def gpu_version(self, sig):
            with cp.cuda.Stream.null:
                out = cusignal.argrelmax(sig)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_argrelmax_cpu(self, rand_2d_data_gen, benchmark, num_samps):
            cpu_sig, _ = rand_2d_data_gen(num_samps)
            benchmark(self.cpu_version, cpu_sig)

        def test_argrelmax_gpu(
            self, rand_2d_data_gen, gpubenchmark, num_samps
        ):

            cpu_sig, gpu_sig = rand_2d_data_gen(num_samps)
            output = gpubenchmark(self.gpu_version, gpu_sig)

            key = self.cpu_version(cpu_sig)
            assert array_equal(cp.asnumpy(output), key)

    @pytest.mark.benchmark(group="Argrelextrema")
    @pytest.mark.parametrize("num_samps", [2 ** 5])
    class TestArgrelextrema:
        def cpu_version(self, sig):
            return signal.argrelextrema(sig, np.less, mode="warp")

        def gpu_version(self, sig):
            with cp.cuda.Stream.null:
                out = cusignal.argrelextrema(sig, np.less)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_argrelextrema_cpu(
            self, rand_2d_data_gen, benchmark, num_samps
        ):
            cpu_sig, _ = rand_2d_data_gen(num_samps)
            benchmark(self.cpu_version, cpu_sig)

        def test_argrelextrema_gpu(
            self, rand_2d_data_gen, gpubenchmark, num_samps
        ):

            cpu_sig, gpu_sig = rand_2d_data_gen(num_samps)
            output = gpubenchmark(self.gpu_version, gpu_sig)

            key = self.cpu_version(cpu_sig)
            assert array_equal(cp.asnumpy(output), key)
