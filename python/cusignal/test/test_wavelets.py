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
from scipy import signal

import cusignal
from cusignal.testing.utils import _check_rapids_pytest_benchmark, array_equal

gpubenchmark = _check_rapids_pytest_benchmark()


class TestWavelets:
    @pytest.mark.benchmark(group="Qmf")
    @pytest.mark.parametrize("num_samps", [2**14])
    class TestQmf:
        def cpu_version(self, sig):
            return signal.qmf(sig)

        def gpu_version(self, sig):
            with cp.cuda.Stream.null:
                out = cusignal.qmf(sig)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_qmf_cpu(self, range_data_gen, benchmark, num_samps):
            cpu_sig, _ = range_data_gen(num_samps)
            benchmark(self.cpu_version, cpu_sig)

        def test_qmf_gpu(self, range_data_gen, gpubenchmark, num_samps):

            cpu_sig, gpu_sig = range_data_gen(num_samps)
            output = gpubenchmark(self.gpu_version, gpu_sig)

            key = self.cpu_version(cpu_sig)
            array_equal(output, key)

    @pytest.mark.benchmark(group="Morlet")
    @pytest.mark.parametrize("num_samps", [2**14])
    class TestMorlet:
        def cpu_version(self, num_samps):
            return signal.morlet(num_samps)

        def gpu_version(self, num_samps):
            with cp.cuda.Stream.null:
                out = cusignal.morlet(num_samps)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_morlet_cpu(self, benchmark, num_samps):
            benchmark(self.cpu_version, num_samps)

        def test_morlet_gpu(self, gpubenchmark, num_samps):

            output = gpubenchmark(self.gpu_version, num_samps)

            key = self.cpu_version(num_samps)
            array_equal(output, key)

    @pytest.mark.benchmark(group="Ricker")
    @pytest.mark.parametrize("num_samps", [2**14])
    @pytest.mark.parametrize("a", [10, 1000])
    class TestRicker:
        def cpu_version(self, num_samps, a):
            return signal.ricker(num_samps, a)

        def gpu_version(self, num_samps, a):
            with cp.cuda.Stream.null:
                out = cusignal.ricker(num_samps, a)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_ricker_cpu(self, benchmark, num_samps, a):
            benchmark(self.cpu_version, num_samps, a)

        def test_ricker_gpu(self, gpubenchmark, num_samps, a):

            output = gpubenchmark(self.gpu_version, num_samps, a)

            key = self.cpu_version(num_samps, a)
            array_equal(output, key)

    @pytest.mark.benchmark(group="Morlet2")
    @pytest.mark.parametrize("num_samps", [2**14])
    @pytest.mark.parametrize("s", [10, 1000])
    class TestMorlet2:
        def cpu_version(self, num_samps, s):
            return signal.morlet2(num_samps, s)

        def gpu_version(self, num_samps, s):
            with cp.cuda.Stream.null:
                out = cusignal.morlet2(num_samps, s)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_morlet2_cpu(self, benchmark, num_samps, s):
            benchmark(self.cpu_version, num_samps, s)

        def test_morlet2_gpu(self, gpubenchmark, num_samps, s):

            output = gpubenchmark(self.gpu_version, num_samps, s)

            key = self.cpu_version(num_samps, s)
            array_equal(output, key)

    @pytest.mark.benchmark(group="CWT")
    @pytest.mark.parametrize("dtype", [np.float64, np.complex128])
    @pytest.mark.parametrize("num_samps", [2**14])
    @pytest.mark.parametrize("widths", [31, 127])
    class TestCWT:
        def cpu_version(self, sig, wavelet, widths):
            return signal.cwt(sig, wavelet, np.arange(1, widths))

        def gpu_version(self, sig, wavelet, widths):
            with cp.cuda.Stream.null:
                out = cusignal.cwt(sig, wavelet, np.arange(1, widths))
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_cwt_cpu(self, rand_data_gen, benchmark, dtype, num_samps, widths):
            cpu_sig, _ = rand_data_gen(num_samps, 1, dtype)
            wavelet = signal.ricker
            benchmark(self.cpu_version, cpu_sig, wavelet, widths)

        def test_cwt_gpu(self, rand_data_gen, gpubenchmark, dtype, num_samps, widths):

            cpu_sig, gpu_sig = rand_data_gen(num_samps, 1, dtype)
            cu_wavelet = cusignal.ricker
            output = gpubenchmark(self.gpu_version, gpu_sig, cu_wavelet, widths)

            wavelet = signal.ricker
            key = self.cpu_version(cpu_sig, wavelet, widths)
            array_equal(output, key)
