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
# rceps
# cceps_unwrap
# cceps


class BenchAcoustics:
    @pytest.mark.benchmark(group="Rceps")
    class BenchRceps:
        def cpu_version(self, cpu_sig):
            return signal.rceps(cpu_sig)

        def bench_rceps_cpu(self, benchmark):
            benchmark(self.cpu_version, cpu_sig)

        def bench_rceps_gpu(self, benchmark):

            output = benchmark(cusignal.detrend, gpu_sig)

            key = self.cpu_version(cpu_sig)
            assert array_equal(cp.asnumpy(output), key)

    @pytest.mark.benchmark(group="CcepsUnwrap")
    class BenchCcepsUnwrap:
        def cpu_version(self, cpu_sig):
            return signal.freq_shift(cpu_sig)

        def bench_cceps_unwrap_cpu(self, benchmark):
            benchmark(self.cpu_version, cpu_sig)

        def bench_cceps_unwrap_gpu(self, benchmark):

            output = benchmark(cusignal.detrend, gpu_sig)

            key = self.cpu_version(cpu_sig)
            assert array_equal(cp.asnumpy(output), key)

    @pytest.mark.benchmark(group="Cceps")
    class BenchCceps:
        def cpu_version(self, cpu_sig):
            return signal.freq_shift(cpu_sig)

        def bench_cceps_cpu(self, benchmark):
            benchmark(self.cpu_version, cpu_sig)

        def bench_cceps_gpu(self, benchmark):

            output = benchmark(cusignal.detrend, gpu_sig)

            key = self.cpu_version(cpu_sig)
            assert array_equal(cp.asnumpy(output), key)
