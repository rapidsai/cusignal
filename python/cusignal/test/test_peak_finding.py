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
# # argrelmax
# # argrelextrema


class TestPeakFinding:
	@pytest.mark.benchmark(group="Argrelmin")
	@pytest.mark.parametrize("num_samps", [2**8])
	@pytest.mark.parametrize("axis", [0, 1])
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
			self, rand_2d_data_gen, benchmark, num_samps, axis, order, mode
		):
			cpu_sig, _ = rand_2d_data_gen(num_samps)
			benchmark(self.cpu_version, cpu_sig, axis, order, mode)

		def test_argrelmin_gpu(
			self, rand_2d_data_gen, gpubenchmark, num_samps, axis, order, mode
		):

			cpu_sig, gpu_sig = rand_2d_data_gen(num_samps)
			output = gpubenchmark(self.gpu_version, gpu_sig, axis, order, mode)

			key = self.cpu_version(cpu_sig, axis, order, mode)
			assert array_equal(cp.asnumpy(output), key)

	# @pytest.mark.benchmark(group="Argrelmin")
	# @pytest.mark.parametrize("x", np.array([[1, 2, 1, 2],
	#                                         [2, 2, 0, 0],
	#                                         [5, 3, 4, 4]]))
	# class TestArgrelmin:
	#     def cpu_version(self, x):
	#         return signal.argrelmin(x)

	#     def gpu_version(self, x):
	#         with cp.cuda.Stream.null:
	#             out = cusignal.argrelmin(x)
	#         cp.cuda.Stream.null.synchronize()
	#         return out

	#     @pytest.mark.cpu
	#     def test_argrelmin_cpu(self, benchmark, x):
	#         benchmark(self.cpu_version, x)

	#     def test_argrelmin_gpu(self, gpubenchmark, x):
	#         d_x = cp.array(x)
	#         #d_x = cp.asarray(x)
	#         output = gpubenchmark(self.gpu_version, d_x)

	#         key = self.cpu_version(x)
	#         assert array_equal(cp.asnumpy(output), key)


#     @pytest.mark.benchmark(group="Argrelmax")
#     class TestArgrelmax:
#         def cpu_version(self, cpu_sig):
#             return signal.argrelmax(cpu_sig)

#         @pytest.mark.cpu
#         def test_argrelmax_cpu(self, benchmark):
#             benchmark(self.cpu_version, cpu_sig)

#         def test_argrelmax_gpu(self, gpubenchmark):

#             output = gpubenchmark(cusignal.argrelmax, gpu_sig)

#             key = self.cpu_version(cpu_sig)
#             assert array_equal(cp.asnumpy(output), key)

#     @pytest.mark.benchmark(group="Argrelextrema")
#     class TestArgrelextrema:
#         def cpu_version(self, cpu_sig):
#             return signal.argrelextrema(cpu_sig)

#         @pytest.mark.cpu
#         def test_argrelextrema_cpu(self, benchmark):
#             benchmark(self.cpu_version, cpu_sig)

#         def test_argrelextrema_gpu(self, gpubenchmark):

#             output = gpubenchmark(cusignal.argrelextrema, gpu_sig)

#             key = self.cpu_version(cpu_sig)
#             assert array_equal(cp.asnumpy(output), key)
