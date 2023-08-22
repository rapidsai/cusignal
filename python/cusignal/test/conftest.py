# Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import cupy as cp
import numpy as np
import pytest

# Fixtures with (scope="session") will execute once
# and be shared will all tests that need it.


def pytest_configure(config):
    config.addinivalue_line("markers", "cpu: mark CPU test cases")


# Generate data for using range
@pytest.fixture(scope="session")
def range_data_gen():
    def _generate(num_samps, endpoint=False):

        cpu_sig = np.arange(num_samps)
        gpu_sig = cp.asarray(cpu_sig)

        return cpu_sig, gpu_sig

    return _generate


# Generate data for using linspace
@pytest.fixture(scope="session")
def linspace_data_gen():
    def _generate(start, stop, num_samps, endpoint=False, dtype=np.float64):

        cpu_time = np.linspace(start, stop, num_samps, endpoint, dtype=dtype)
        cpu_sig = np.cos(-(cpu_time**2) / 6.0)
        gpu_sig = cp.asarray(cpu_sig)

        return cpu_sig, gpu_sig

    return _generate


# Generate data for using linspace
@pytest.fixture(scope="session")
def linspace_range_gen():
    def _generate(num_samps):

        cpu_sig = np.arange(num_samps) / num_samps
        gpu_sig = cp.asarray(cpu_sig)

        return cpu_sig, gpu_sig

    return _generate


# Generate array with random data
@pytest.fixture(scope="session")
def rand_data_gen():
    def _generate(num_samps, dim=1, dtype=np.float64):

        if dtype is np.float32 or dtype is np.float64:
            inp = tuple(np.ones(dim, dtype=int) * num_samps)
            cpu_sig = np.random.random(inp)
            cpu_sig = cpu_sig.astype(dtype)
            gpu_sig = cp.asarray(cpu_sig)
        else:
            inp = tuple(np.ones(dim, dtype=int) * num_samps)
            cpu_sig = np.random.random(inp) + 1j * np.random.random(inp)
            cpu_sig = cpu_sig.astype(dtype)
            gpu_sig = cp.asarray(cpu_sig)

        return cpu_sig, gpu_sig

    return _generate


# Generate time array with linspace
@pytest.fixture(scope="session")
def time_data_gen():
    def _generate(start, stop, num_samps):

        cpu_sig = np.linspace(start, stop, num_samps)
        gpu_sig = cp.asarray(cpu_sig)

        return cpu_sig, gpu_sig

    return _generate


# Generate input for lombscargle
@pytest.fixture(scope="session")
def lombscargle_gen(rand_data_gen):
    def _generate(num_in_samps, num_out_samps):

        A = 2.0
        w = 1.0
        phi = 0.5 * np.pi
        frac_points = 0.9  # Fraction of points to select

        r, _ = rand_data_gen(num_in_samps, 1)
        cpu_x = np.linspace(0.01, 10 * np.pi, num_in_samps)

        cpu_x = cpu_x[r >= frac_points]

        cpu_y = A * np.cos(w * cpu_x + phi)

        cpu_f = np.linspace(0.01, 10, num_out_samps)

        gpu_x = cp.asarray(cpu_x)
        gpu_y = cp.asarray(cpu_y)
        gpu_f = cp.asarray(cpu_f)

        return cpu_x, cpu_y, cpu_f, gpu_x, gpu_y, gpu_f

    return _generate
