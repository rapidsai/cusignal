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

from cusignal.test.utils import array_equal
from scipy import signal

# Missing
# kaiser_beta
# kaiser_atten
# cmplx_sort


class TestFilterDesign:
    @pytest.mark.parametrize("num_samps", [2 ** 15])
    @pytest.mark.parametrize("f1", [0.1, 0.15])
    @pytest.mark.parametrize("f2", [0.2, 0.4])
    def test_firwin(self, num_samps, f1, f2):
        cpu_window = signal.firwin(num_samps, [f1, f2], pass_zero=False)
        gpu_window = cp.asnumpy(
            cusignal.firwin(num_samps, [f1, f2], pass_zero=False)
        )
        assert array_equal(cpu_window, gpu_window)

    def test_kaiser_beta(self):
        cpu_window = 0
        gpu_window = 0
        assert array_equal(cpu_window, gpu_window)

    def test_kaiser_atten(self):
        cpu_window = 0
        gpu_window = 0
        assert array_equal(cpu_window, gpu_window)

    def test_cmplx_sort(self):
        cpu_window = 0
        gpu_window = 0
        assert array_equal(cpu_window, gpu_window)
