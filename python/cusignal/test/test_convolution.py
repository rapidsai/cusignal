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


class TestConvolution:
    @pytest.mark.parametrize("num_samps", [2 ** 7, 1025, 2 ** 15])
    @pytest.mark.parametrize("num_taps", [125, 2 ** 8, 2 ** 15])
    @pytest.mark.parametrize("mode", ["full", "valid", "same"])
    @pytest.mark.parametrize("method", ["direct", "fft", "auto"])
    def test_correlate(self, rand_data_gen, num_samps, num_taps, mode, method):
        cpu_sig, gpu_sig = rand_data_gen(num_samps)

        cpu_corr = signal.correlate(
            cpu_sig, np.ones(num_taps), mode=mode, method=method
        )
        gpu_corr = cp.asnumpy(
            cusignal.correlate(
                gpu_sig, cp.ones(num_taps), mode=mode, method=method
            )
        )
        assert array_equal(cpu_corr, gpu_corr)

    @pytest.mark.parametrize("num_samps", [2 ** 7, 1025, 2 ** 15])
    @pytest.mark.parametrize("num_taps", [125, 2 ** 8, 2 ** 15])
    @pytest.mark.parametrize("mode", ["full", "valid", "same"])
    @pytest.mark.parametrize("method", ["direct", "fft", "auto"])
    def test_convolve(self, num_samps, num_taps, mode, method):
        cpu_sig = np.random.rand(num_samps)
        cpu_win = signal.windows.hann(num_taps)

        gpu_sig = cp.asarray(cpu_sig)
        gpu_win = cusignal.windows.hann(num_taps)

        cpu_conv = signal.convolve(cpu_sig, cpu_win, mode=mode, method=method)
        gpu_conv = cp.asnumpy(
            cusignal.convolve(gpu_sig, gpu_win, mode=mode, method=method)
        )
        assert array_equal(cpu_conv, gpu_conv)

    @pytest.mark.parametrize("num_samps", [2 ** 15])
    def test_fftconvolve(self, num_samps, mode="full"):
        cpu_sig = np.random.rand(num_samps)
        gpu_sig = cp.asarray(cpu_sig)

        cpu_autocorr = signal.fftconvolve(cpu_sig, cpu_sig[::-1], mode=mode)
        gpu_autocorr = cp.asnumpy(
            cusignal.fftconvolve(gpu_sig, gpu_sig[::-1], mode=mode)
        )
        assert array_equal(cpu_autocorr, gpu_autocorr)

    @pytest.mark.parametrize("num_samps", [2 ** 8])
    @pytest.mark.parametrize("num_taps", [5, 100])
    @pytest.mark.parametrize("boundary", ["symm"])
    @pytest.mark.parametrize("mode", ["same"])
    def test_convolve2d(self, num_samps, num_taps, boundary, mode):
        cpu_sig = np.random.rand(num_samps, num_samps)
        cpu_filt = np.random.rand(num_taps, num_taps)
        gpu_sig = cp.asarray(cpu_sig)
        gpu_filt = cp.asarray(cpu_filt)

        cpu_convolve2d = signal.convolve2d(
            cpu_sig, cpu_filt, boundary=boundary, mode=mode
        )

        gpu_convolve2d = cp.asnumpy(
            cusignal.convolve2d(
                gpu_sig, gpu_filt, boundary=boundary, mode=mode,
            )
        )
        assert array_equal(cpu_convolve2d, gpu_convolve2d)

    @pytest.mark.parametrize("num_samps", [2 ** 8])
    @pytest.mark.parametrize("num_taps", [5, 100])
    @pytest.mark.parametrize("boundary", ["symm"])
    @pytest.mark.parametrize("mode", ["same"])
    def test_correlate2d(
        self, rand_2d_data_gen, num_samps, num_taps, boundary, mode
    ):
        cpu_sig, gpu_sig = rand_2d_data_gen(num_samps)
        cpu_filt, gpu_filt = rand_2d_data_gen(num_taps)

        cpu_correlate2d = signal.correlate2d(
            cpu_sig, cpu_filt, boundary=boundary, mode=mode
        )
        gpu_correlate2d = cp.asnumpy(
            cusignal.correlate2d(
                gpu_sig, gpu_filt, boundary=boundary, mode=mode,
            )
        )
        assert array_equal(cpu_correlate2d, gpu_correlate2d)
