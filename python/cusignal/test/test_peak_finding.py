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

# from cusignal.test.utils import array_equal
# from scipy import signal

# # Missing
# # argrelmin
# # argrelmax
# # argrelextrema


# class TestPeakFinding:
#     @pytest.mark.benchmark(group="Argrelmin")
#     class TestArgrelmin:
#         def cpu_version(self, cpu_sig):
#             return signal.argrelmin(cpu_sig)

#         @pytest.mark.cpu
#         def test_argrelmin_cpu(self, benchmark):
#             benchmark(self.cpu_version, cpu_sig)

#         def test_argrelmin_gpu(self, benchmark):

#             output = benchmark(cusignal.argrelmin, gpu_sig)

#             key = self.cpu_version(cpu_sig)
#             assert array_equal(cp.asnumpy(output), key)

#     @pytest.mark.benchmark(group="Argrelmax")
#     class TestArgrelmax:
#         def cpu_version(self, cpu_sig):
#             return signal.argrelmax(cpu_sig)

#         @pytest.mark.cpu
#         def test_argrelmax_cpu(self, benchmark):
#             benchmark(self.cpu_version, cpu_sig)

#         def test_argrelmax_gpu(self, benchmark):

#             output = benchmark(cusignal.argrelmax, gpu_sig)

#             key = self.cpu_version(cpu_sig)
#             assert array_equal(cp.asnumpy(output), key)

#     @pytest.mark.benchmark(group="Argrelextrema")
#     class TestArgrelextrema:
#         def cpu_version(self, cpu_sig):
#             return signal.argrelextrema(cpu_sig)

#         @pytest.mark.cpu
#         def test_argrelextrema_cpu(self, benchmark):
#             benchmark(self.cpu_version, cpu_sig)

#         def test_argrelextrema_gpu(self, benchmark):

#             output = benchmark(cusignal.argrelextrema, gpu_sig)

#             key = self.cpu_version(cpu_sig)
#             assert array_equal(cp.asnumpy(output), key)
