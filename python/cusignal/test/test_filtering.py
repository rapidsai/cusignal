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

# Missing
# lfiltic
# detrend
# freq_shift


class TestFiltering:
    @pytest.mark.parametrize("num_samps", [2 ** 15, 2 ** 24])
    def test_wiener(self, num_samps):
        cpu_sig = np.random.rand(num_samps)
        gpu_sig = cp.asarray(cpu_sig)

        cpu_wfilt = signal.wiener(cpu_sig)
        gpu_wfilt = cp.asnumpy(cusignal.wiener(gpu_sig))
        assert array_equal(cpu_wfilt, gpu_wfilt)

    def test_lfiltic(self):
        cpu_window = 0
        gpu_window = 0
        assert array_equal(cpu_window, gpu_window)

    @pytest.mark.parametrize("num_signals", [1, 2, 10])
    @pytest.mark.parametrize("num_samps", [100])
    def test_sosfilt(self, num_signals, num_samps):
        cpu_sig = np.random.rand(num_signals, num_samps)
        gpu_sig = cp.asarray(cpu_sig)

        cpu_sos = signal.ellip(64, 0.009, 80, 0.05, output="sos")

        cpu_sosfilt = signal.sosfilt(cpu_sos, cpu_sig)

        gpu_sos = cp.asarray(cpu_sos)

        gpu_sosfilt = cp.asnumpy(cusignal.sosfilt(gpu_sos, gpu_sig))

        assert array_equal(cpu_sosfilt, gpu_sosfilt)

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

    def test_detrend(self):
        cpu_window = 0
        gpu_window = 0
        assert array_equal(cpu_window, gpu_window)

    def test_freq_shift(self):
        cpu_window = 0
        gpu_window = 0
        assert array_equal(cpu_window, gpu_window)

    @pytest.mark.parametrize("num_samps", [2 ** 14])
    @pytest.mark.parametrize("downsample_factor", [2, 3, 4, 8, 64])
    @pytest.mark.parametrize("zero_phase", [True, False])
    def test_decimate(
        self, linspace_data_gen, num_samps, downsample_factor, zero_phase
    ):
        cpu_sig, gpu_sig = linspace_data_gen(0, 10, num_samps, endpoint=False)

        cpu_decimate = signal.decimate(
            cpu_sig, downsample_factor, ftype="fir", zero_phase=zero_phase
        )
        gpu_decimate = cp.asnumpy(
            cusignal.decimate(
                gpu_sig, downsample_factor, zero_phase=zero_phase
            )
        )

        assert array_equal(cpu_decimate, gpu_decimate)

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
            cusignal.resample_poly(gpu_sig, up, down, window=window)
        )

        assert array_equal(cpu_resample, gpu_resample)

    @pytest.mark.parametrize("num_samps", [2 ** 14])
    @pytest.mark.parametrize("up", [2, 3, 7])
    @pytest.mark.parametrize("down", [1, 2, 9])
    def test_upfirdn(self, rand_data_gen, num_samps, up, down):
        cpu_sig, gpu_sig = rand_data_gen(num_samps)

        h = [1, 1, 1]

        cpu_resample = signal.upfirdn(h, cpu_sig, up, down)
        gpu_resample = cp.asnumpy(cusignal.upfirdn(h, gpu_sig, up, down))

        assert array_equal(cpu_resample, gpu_resample)

    @pytest.mark.parametrize("num_samps", [2 ** 8])
    @pytest.mark.parametrize("up", [2, 3, 7])
    @pytest.mark.parametrize("down", [1, 2, 9])
    def test_upfirdn2d(self, rand_2d_data_gen, num_samps, up, down):
        cpu_sig, gpu_sig = rand_2d_data_gen(num_samps)

        h = [1, 1, 1]

        cpu_resample = signal.upfirdn(h, cpu_sig, up, down)
        gpu_resample = cp.asnumpy(cusignal.upfirdn(h, gpu_sig, up, down))

        assert array_equal(cpu_resample, gpu_resample)
