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

# import cupy as cp
# import cusignal
# import numpy as np
# import pytest
# import pytest_benchmark

# from cusignal.test.utils import array_equal
# from scipy import signal

# try:
#     from rapids_pytest_benchmark import setFixtureParamNames
# except ImportError:
#     print("\n\nWARNING: rapids_pytest_benchmark is not installed, "
#           "falling back to pytest_benchmark fixtures.\n")

#     # if rapids_pytest_benchmark is not available, just perfrom time-only
#     # benchmarking and replace the util functions with nops
#     gpubenchmark = pytest_benchmark.plugin.benchmark

#     def setFixtureParamNames(*args, **kwargs):
#         pass

# # Missing
# # rceps
# # cceps_unwrap
# # cceps


# class TestAcoustics:
#     @pytest.mark.benchmark(group="Rceps")
#     class TestRceps:
#         def cpu_version(self, cpu_sig):
#             return signal.rceps(cpu_sig)

#         @pytest.mark.cpu
#         def test_rceps_cpu(self, benchmark):
#             benchmark(self.cpu_version, cpu_sig)

#         def test_rceps_gpu(self, gpubenchmark):

#             output = gpubenchmark(cusignal.detrend, gpu_sig)

#             key = self.cpu_version(cpu_sig)
#             assert array_equal(cp.asnumpy(output), key)

#     @pytest.mark.benchmark(group="CcepsUnwrap")
#     class TestCcepsUnwrap:
#         def cpu_version(self, cpu_sig):
#             return signal.freq_shift(cpu_sig)

#         @pytest.mark.cpu
#         def test_cceps_unwrap_cpu(self, benchmark):
#             benchmark(self.cpu_version, cpu_sig)

#         def test_cceps_unwrap_gpu(self, gpubenchmark):

#             output = gpubenchmark(cusignal.detrend, gpu_sig)

#             key = self.cpu_version(cpu_sig)
#             assert array_equal(cp.asnumpy(output), key)

#     @pytest.mark.benchmark(group="Cceps")
#     class TestCceps:
#         def cpu_version(self, cpu_sig):
#             return signal.freq_shift(cpu_sig)

#         @pytest.mark.cpu
#         def test_cceps_cpu(self, benchmark):
#             benchmark(self.cpu_version, cpu_sig)

#         def test_cceps_gpu(self, gpubenchmark):

#             output = gpubenchmark(cusignal.detrend, gpu_sig)

#             key = self.cpu_version(cpu_sig)
#             assert array_equal(cp.asnumpy(output), key)
