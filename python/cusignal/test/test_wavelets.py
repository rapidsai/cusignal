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
import numpy as np
import pytest
from scipy import signal

import cusignal
from cusignal.test.utils import array_equal


@pytest.mark.parametrize("num_samps", [2 ** 14])
def test_morlet(num_samps):
    cpu_window = signal.morlet(num_samps)
    gpu_window = cp.asnumpy(cusignal.morlet(num_samps))
    assert array_equal(cpu_window, gpu_window)


@pytest.mark.parametrize("num_samps", [2 ** 14])
@pytest.mark.parametrize("a", [10, 1000])
def test_ricker(num_samps, a):
    cpu_window = signal.ricker(num_samps, a)
    gpu_window = cp.asnumpy(cusignal.ricker(num_samps, a))
    assert array_equal(cpu_window, gpu_window)


@pytest.mark.parametrize("num_samps", [2 ** 14])
@pytest.mark.parametrize("widths", [31, 127])
def test_cwt(num_samps, widths):
    cpu_signal = np.random.rand(int(num_samps))
    gpu_signal = cp.asarray(cpu_signal)

    cpu_cwt = signal.cwt(cpu_signal, signal.ricker, np.arange(1, widths))
    gpu_cwt = cp.asnumpy(
        cusignal.cwt(gpu_signal, cusignal.ricker, cp.arange(1, widths))
    )

    assert array_equal(cpu_cwt, gpu_cwt)


@pytest.mark.parametrize("num_samps", [2 ** 14])
@pytest.mark.parametrize("widths", [31, 127])
def test_cwt_complex(num_samps, widths):
    cpu_signal = np.random.rand(int(num_samps)) + 1j * np.random.rand(
        int(num_samps)
    )
    gpu_signal = cp.asarray(cpu_signal)

    cpu_cwt = signal.cwt(cpu_signal, signal.ricker, np.arange(1, widths))
    gpu_cwt = cp.asnumpy(
        cusignal.cwt(gpu_signal, cusignal.ricker, cp.arange(1, widths))
    )

    assert array_equal(cpu_cwt, gpu_cwt)
