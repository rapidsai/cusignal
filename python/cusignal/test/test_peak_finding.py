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


class TestPeakFinding:
    @pytest.mark.benchmark(group="Argrelmin")
    @pytest.mark.parametrize(
        "dim, num_samps", [(1, 2 ** 15), (2, 2 ** 8), (3, 2 ** 5)]
    )
    @pytest.mark.parametrize("axis", [-1])
    @pytest.mark.parametrize("order", [1, 2])
    @pytest.mark.parametrize("mode", ["clip", "wrap"])
    class TestArgrelmin:
        def cpu_version(self, sig, axis, order, mode):
            return signal.argrelmin(sig, axis, order, mode)

        def gpu_version(self, sig, axis, order, mode):
            with cp.cuda.Stream.null:
                out = cusignal.argrelmin(sig, axis, order, mode)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_argrelmin_cpu(
            self, rand_data_gen, benchmark, dim, num_samps, axis, order, mode
        ):
            cpu_sig, _ = rand_data_gen(num_samps, dim)
            benchmark(self.cpu_version, cpu_sig, axis, order, mode)

        def test_argrelmin_gpu(
            self,
            rand_data_gen,
            gpubenchmark,
            dim,
            num_samps,
            axis,
            order,
            mode,
        ):
            cpu_sig, gpu_sig = rand_data_gen(num_samps, dim)
            output = gpubenchmark(self.gpu_version, gpu_sig, axis, order, mode)
            key = self.cpu_version(cpu_sig, axis, order, mode)
            array_equal(output, key)

    @pytest.mark.benchmark(group="TestArgrelmax")
    @pytest.mark.parametrize(
        "dim, num_samps", [(1, 2 ** 15), (2, 2 ** 8), (3, 2 ** 5)]
    )
    @pytest.mark.parametrize("axis", [-1])
    @pytest.mark.parametrize("order", [1, 2])
    @pytest.mark.parametrize("mode", ["clip", "wrap"])
    class TestArgrelmax:
        def cpu_version(self, sig, axis, order, mode):
            return signal.argrelmax(sig, axis, order, mode)

        def gpu_version(self, sig, axis, order, mode):
            with cp.cuda.Stream.null:
                out = cusignal.argrelmax(sig, axis, order, mode)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_argrelmax_cpu(
            self, rand_data_gen, benchmark, dim, num_samps, axis, order, mode
        ):
            cpu_sig, _ = rand_data_gen(num_samps, dim)
            benchmark(self.cpu_version, cpu_sig, axis, order, mode)

        def test_argrelmax_gpu(
            self,
            rand_data_gen,
            gpubenchmark,
            dim,
            num_samps,
            axis,
            order,
            mode,
        ):
            cpu_sig, gpu_sig = rand_data_gen(num_samps, dim)
            output = gpubenchmark(self.gpu_version, gpu_sig, axis, order, mode)
            key = self.cpu_version(cpu_sig, axis, order, mode)
            array_equal(output, key)

    @pytest.mark.benchmark(group="Argrelextrema")
    @pytest.mark.parametrize(
        "dim, num_samps", [(1, 2 ** 15), (2, 2 ** 8), (3, 2 ** 5)]
    )
    @pytest.mark.parametrize("axis", [-1])
    @pytest.mark.parametrize("order", [1, 2])
    @pytest.mark.parametrize("mode", ["clip", "wrap"])
    class TestArgrelextrema:
        def cpu_version(self, sig, axis, order, mode):
            return signal.argrelextrema(sig, np.less, axis, order, mode)

        def gpu_version(self, sig, axis, order, mode):
            with cp.cuda.Stream.null:
                out = cusignal.argrelextrema(sig, cp.less, axis, order, mode)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_argrelextrema_cpu(
            self,
            rand_data_gen,
            benchmark,
            dim,
            num_samps,
            axis,
            order,
            mode,
        ):
            cpu_sig, _ = rand_data_gen(num_samps, dim)
            benchmark(self.cpu_version, cpu_sig, axis, order, mode)

        def test_argrelextrema_gpu(
            self,
            rand_data_gen,
            gpubenchmark,
            dim,
            num_samps,
            axis,
            order,
            mode,
        ):
            cpu_sig, gpu_sig = rand_data_gen(num_samps, dim)
            output = gpubenchmark(self.gpu_version, gpu_sig, axis, order, mode)
            key = self.cpu_version(cpu_sig, axis, order, mode)
            array_equal(output, key)
