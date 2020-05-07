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


class TestWaveforms:
    @pytest.mark.parametrize('num_samps', [2**14])
    @pytest.mark.parametrize('duty', [0.25, 0.5])
    def test_square(self, time_data_gen, num_samps, duty):
        cpu_time, gpu_time = time_data_gen(0, 10, num_samps)

        cpu_pwm = signal.square(cpu_time, duty)
        gpu_pwm = cp.asnumpy(cusignal.square(gpu_time, duty))

        assert array_equal(cpu_pwm, gpu_pwm)

    @pytest.mark.parametrize('num_samps', [2**14])
    @pytest.mark.parametrize('fc', [0.75, 5])
    def test_gausspulse(self, time_data_gen, num_samps, fc):
        cpu_time, gpu_time = time_data_gen(0, 10, num_samps)

        cpu_pwm = signal.gausspulse(cpu_time, fc, retquad=True, retenv=True)
        gpu_pwm = cp.asnumpy(
            cusignal.gausspulse(gpu_time, fc, retquad=True, retenv=True)
        )

        assert array_equal(cpu_pwm, gpu_pwm)

    @pytest.mark.parametrize('num_samps', [2**14])
    @pytest.mark.parametrize('f0', [6])
    @pytest.mark.parametrize('t1', [1])
    @pytest.mark.parametrize('f1', [10])
    @pytest.mark.parametrize('method', ['linear', 'quadratic'])
    def test_chirp(self, time_data_gen, num_samps, f0, t1, f1, method):
        cpu_time, gpu_time = time_data_gen(0, 10, num_samps)

        cpu_chirp = signal.chirp(cpu_time, f0, t1, f1, method)
        gpu_chirp = cp.asnumpy(cusignal.chirp(gpu_time, f0, t1, f1, method))

        assert array_equal(cpu_chirp, gpu_chirp)

    @pytest.mark.parametrize('num_samps', [2**14])
    @pytest.mark.parametrize('idx', ['mid'])
    def test_unit_impulse(self, num_samps, idx):
        cpu_uimp = signal.unit_impulse(num_samps, idx)
        gpu_uimp = cp.asnumpy(cusignal.unit_impulse(num_samps, idx))

        assert array_equal(cpu_uimp, gpu_uimp)
