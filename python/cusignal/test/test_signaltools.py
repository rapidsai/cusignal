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


class TestSignaltools:
    @pytest.mark.parametrize("num_samps", [2 ** 14])
    @pytest.mark.parametrize("resample_num_samps", [2 ** 12, 2 ** 16])
    @pytest.mark.parametrize("window", [("kaiser", 0.5)])
    def test_resample(
        self, linspace_data_gen, num_samps, resample_num_samps, window
    ):
        cpu_sig, gpu_sig = linspace_data_gen(0, 10, num_samps, endpoint=False)

        cpu_resample = signal.resample(
            cpu_sig, resample_num_samps, window=window
        )
        gpu_resample = cp.asnumpy(
            cusignal.resample(gpu_sig, resample_num_samps, window=window)
        )

        assert array_equal(cpu_resample, gpu_resample)

    @pytest.mark.parametrize("num_samps", [2 ** 14])
    @pytest.mark.parametrize("up", [2, 3, 7])
    @pytest.mark.parametrize("down", [1, 2, 9])
    @pytest.mark.parametrize("window", [("kaiser", 0.5)])
    def test_resample_poly(
        self, linspace_data_gen, num_samps, up, down, window
    ):
        cpu_sig, gpu_sig = linspace_data_gen(0, 10, num_samps, endpoint=False)

        cpu_resample = signal.resample_poly(cpu_sig, up, down, window=window)
        gpu_resample = cp.asnumpy(
            cusignal.resample_poly(
                gpu_sig, up, down, window=window
            )
        )

        assert array_equal(cpu_resample, gpu_resample)

    @pytest.mark.parametrize("num_samps", [2 ** 14])
    @pytest.mark.parametrize("up", [2, 3, 7])
    @pytest.mark.parametrize("down", [1, 2, 9])
    def test_upfirdn(self, rand_data_gen, num_samps, up, down):
        cpu_sig, gpu_sig = rand_data_gen(num_samps)

        h = [1, 1, 1]

        cpu_resample = signal.upfirdn(h, cpu_sig, up, down)
        gpu_resample = cp.asnumpy(
            cusignal.upfirdn(h, gpu_sig, up, down)
        )

        assert array_equal(cpu_resample, gpu_resample)

    @pytest.mark.parametrize("num_samps", [2 ** 8])
    @pytest.mark.parametrize("up", [2, 3, 7])
    @pytest.mark.parametrize("down", [1, 2, 9])
    def test_upfirdn2d(self, rand_2d_data_gen, num_samps, up, down):
        cpu_sig, gpu_sig = rand_2d_data_gen(num_samps)

        h = [1, 1, 1]

        cpu_resample = signal.upfirdn(h, cpu_sig, up, down)
        gpu_resample = cp.asnumpy(
            cusignal.upfirdn(h, gpu_sig, up, down)
        )

        assert array_equal(cpu_resample, gpu_resample)

    @pytest.mark.parametrize("num_samps", [2 ** 15])
    @pytest.mark.parametrize("f1", [0.1, 0.15])
    @pytest.mark.parametrize("f2", [0.2, 0.4])
    def test_firwin(self, num_samps, f1, f2):
        cpu_window = signal.firwin(num_samps, [f1, f2], pass_zero=False)
        gpu_window = cp.asnumpy(
            cusignal.firwin(num_samps, [f1, f2], pass_zero=False)
        )
        assert array_equal(cpu_window, gpu_window)

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

    @pytest.mark.parametrize("num_samps", [2 ** 15, 2 ** 24])
    def test_wiener(self, num_samps):
        cpu_sig = np.random.rand(num_samps)
        gpu_sig = cp.asarray(cpu_sig)

        cpu_wfilt = signal.wiener(cpu_sig)
        gpu_wfilt = cp.asnumpy(cusignal.wiener(gpu_sig))
        assert array_equal(cpu_wfilt, gpu_wfilt)

    @pytest.mark.parametrize("num_samps", [2 ** 15])
    def test_hilbert(self, num_samps):
        cpu_sig = np.random.rand(num_samps)
        gpu_sig = cp.asarray(cpu_sig)

        cpu_hilbert = signal.hilbert(cpu_sig)
        gpu_hilbert = cp.asnumpy(cusignal.hilbert(gpu_sig))
        assert array_equal(cpu_hilbert, gpu_hilbert)

    @pytest.mark.parametrize("num_samps", [2 ** 8])
    def test_hilbert2(self, num_samps):
        cpu_sig = np.random.rand(num_samps, num_samps)
        gpu_sig = cp.asarray(cpu_sig)

        cpu_hilbert2 = signal.hilbert2(cpu_sig)
        gpu_hilbert2 = cp.asnumpy(cusignal.hilbert2(gpu_sig))
        assert array_equal(cpu_hilbert2, gpu_hilbert2)

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
                gpu_sig,
                gpu_filt,
                boundary=boundary,
                mode=mode,
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
                gpu_sig,
                gpu_filt,
                boundary=boundary,
                mode=mode,
            )
        )
        assert array_equal(cpu_correlate2d, gpu_correlate2d)

    @pytest.mark.parametrize("num_samps", [2 ** 14])
    @pytest.mark.parametrize("downsample_factor", [2, 3, 4, 8, 64])
    @pytest.mark.parametrize("zero_phase", [True, False])
    def test_decimate(self, num_samps, downsample_factor, zero_phase):
        cpu_time = np.linspace(0, 10, num_samps, endpoint=False)
        cpu_sig = np.cos(-(cpu_time ** 2) / 6.0)
        gpu_sig = cp.asarray(cpu_sig)

        cpu_decimate = signal.decimate(
            cpu_sig, downsample_factor, ftype="fir", zero_phase=zero_phase
        )
        gpu_decimate = cp.asnumpy(
            cusignal.decimate(
                gpu_sig, downsample_factor, zero_phase=zero_phase
            )
        )

        assert array_equal(cpu_decimate, gpu_decimate)
