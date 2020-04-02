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

import pytest
import cupy as cp
import numpy as np

# Fixtures with (scope="session") will execute once
# and be shared will all tests that need it.

# Generate data for resample and resample_poly
@pytest.fixture(scope="session")
def resample_data_gen():
    def _generate(arg):

        cpu_time = np.linspace(0, 10, endpoint=False)
        cpu_sig = np.cos(-(cpu_time ** 2) / 6.0)
        gpu_sig = cp.asarray(cpu_sig)

        return cpu_sig, gpu_sig

    return _generate


# Generate array with random data
@pytest.fixture(scope="session")
def rand_data_gen():
    def _generate(arg):

        cpu_sig = np.random.rand(arg)
        gpu_sig = cp.asarray(cpu_sig)

        return cpu_sig, gpu_sig

    return _generate


# Generate array with random complex data
@pytest.fixture(scope="session")
def rand_complex_data_gen():
    def _generate(arg):

        cpu_sig = np.random.rand(arg) + 1j * np.random.rand(arg)
        gpu_sig = cp.asarray(cpu_sig)

        return cpu_sig, gpu_sig

    return _generate


# Generate 2d array with random data
@pytest.fixture(scope="session")
def rand_2d_data_gen():
    def _generate(arg):

        cpu_sig = np.random.rand(arg, arg)
        gpu_sig = cp.asarray(cpu_sig)

        return cpu_sig, gpu_sig

    return _generate
