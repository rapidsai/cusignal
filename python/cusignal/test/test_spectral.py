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


class TestSpectral:
    @pytest.mark.parametrize("num_samps", [2 ** 14])
    @pytest.mark.parametrize("fs", [1.0, 1e6])
    @pytest.mark.parametrize("nperseg", [1024, 2048])
    def test_csd(self, rand_data_gen, num_samps, fs, nperseg):
        cpu_x, gpu_x = rand_data_gen(num_samps)
        cpu_y, gpu_y = rand_data_gen(num_samps)

        cpu_csd = signal.csd(cpu_x, cpu_y, fs, nperseg=nperseg)
        gpu_csd = cp.asnumpy(cusignal.csd(gpu_x, gpu_y, fs, nperseg=nperseg))

        assert array_equal(cpu_csd, gpu_csd)

    @pytest.mark.parametrize("num_samps", [2 ** 14])
    @pytest.mark.parametrize("fs", [1.0, 1e6])
    @pytest.mark.parametrize("nperseg", [1024, 2048])
    def test_csd_complex(self, rand_complex_data_gen, num_samps, fs, nperseg):
        cpu_x, gpu_x = rand_complex_data_gen(num_samps)
        cpu_y, gpu_y = rand_complex_data_gen(num_samps)

        cpu_csd = signal.csd(cpu_x, cpu_y, fs, nperseg=nperseg)
        gpu_csd = cp.asnumpy(cusignal.csd(gpu_x, gpu_y, fs, nperseg=nperseg))

        assert array_equal(cpu_csd, gpu_csd)

    @pytest.mark.parametrize("num_samps", [2 ** 14])
    @pytest.mark.parametrize("fs", [1.0, 1e6])
    @pytest.mark.parametrize("window", ["flattop", "nuttall"])
    @pytest.mark.parametrize("scaling", ["spectrum", "density"])
    def test_periodogram(self, rand_data_gen, num_samps, fs, window, scaling):
        cpu_sig, gpu_sig = rand_data_gen(num_samps)

        cpu_periodogram = signal.periodogram(
            cpu_sig, fs, window=window, scaling=scaling
        )
        gpu_periodogram = cp.asnumpy(
            cusignal.periodogram(gpu_sig, fs, window=window, scaling=scaling)
        )

        assert array_equal(cpu_periodogram, gpu_periodogram)

    @pytest.mark.parametrize("num_samps", [2 ** 14])
    @pytest.mark.parametrize("fs", [1.0, 1e6])
    @pytest.mark.parametrize("window", ["flattop", "nuttall"])
    @pytest.mark.parametrize("scaling", ["spectrum", "density"])
    def test_periodogram_complex(
        self, rand_complex_data_gen, num_samps, fs, window, scaling
    ):
        cpu_sig, gpu_sig = rand_complex_data_gen(num_samps)

        _, cpu_periodogram = signal.periodogram(
            cpu_sig, fs, window=window, scaling=scaling
        )
        _, gpu_periodogram = cusignal.periodogram(
            gpu_sig, fs, window=window, scaling=scaling
        )
        gpu_periodogram = cp.asnumpy(gpu_periodogram)

        assert array_equal(cpu_periodogram, gpu_periodogram)

    @pytest.mark.parametrize("num_samps", [2 ** 14])
    @pytest.mark.parametrize("fs", [1.0, 1e6])
    @pytest.mark.parametrize("nperseg", [1024, 2048])
    def test_welch(self, rand_data_gen, num_samps, fs, nperseg):
        cpu_sig, gpu_sig = rand_data_gen(num_samps)

        _, cPxx_spec = signal.welch(cpu_sig, fs, nperseg=nperseg)
        _, gPxx_spec = cusignal.welch(gpu_sig, fs, nperseg=nperseg)
        gPxx_spec = cp.asnumpy(gPxx_spec)

        assert array_equal(cPxx_spec, gPxx_spec)

    @pytest.mark.parametrize("num_samps", [2 ** 14])
    @pytest.mark.parametrize("fs", [1.0, 1e6])
    @pytest.mark.parametrize("nperseg", [1024, 2048])
    def test_welch_complex(
        self, rand_complex_data_gen, num_samps, fs, nperseg
    ):
        cpu_sig, gpu_sig = rand_complex_data_gen(num_samps)

        _, cPxx_spec = signal.welch(cpu_sig, fs, nperseg=nperseg)
        _, gPxx_spec = cusignal.welch(gpu_sig, fs, nperseg=nperseg)
        gPxx_spec = cp.asnumpy(gPxx_spec)

        assert array_equal(cPxx_spec, gPxx_spec)

    @pytest.mark.parametrize("num_samps", [2 ** 14])
    @pytest.mark.parametrize("fs", [1.0, 1e6])
    def test_spectrogram(self, rand_data_gen, num_samps, fs):
        cpu_sig, gpu_sig = rand_data_gen(num_samps)

        _, _, cPxx_spec = signal.spectrogram(cpu_sig, fs)
        _, _, gPxx_spec = cusignal.spectrogram(gpu_sig, fs)
        gPxx_spec = cp.asnumpy(gPxx_spec)

        assert array_equal(cPxx_spec, gPxx_spec)

    @pytest.mark.parametrize("num_samps", [2 ** 14])
    @pytest.mark.parametrize("fs", [1.0, 1e6])
    def test_spectrogram_complex(self, rand_complex_data_gen, num_samps, fs):
        cpu_sig, gpu_sig = rand_complex_data_gen(num_samps)

        _, _, cPxx_spec = signal.spectrogram(cpu_sig, fs)
        _, _, gPxx_spec = cusignal.spectrogram(gpu_sig, fs)
        gPxx_spec = cp.asnumpy(gPxx_spec)

        assert array_equal(cPxx_spec, gPxx_spec)

    @pytest.mark.parametrize("num_samps", [2 ** 14])
    @pytest.mark.parametrize("fs", [1.0, 1e6])
    @pytest.mark.parametrize("nperseg", [1024, 2048])
    def test_coherence(self, rand_data_gen, num_samps, fs, nperseg):
        cpu_x, gpu_x = rand_data_gen(num_samps)
        cpu_y, gpu_y = rand_data_gen(num_samps)

        _, cpu_coherence = signal.coherence(cpu_x, cpu_y, fs, nperseg=nperseg)
        _, gpu_coherence = cusignal.coherence(
            gpu_x, gpu_y, fs, nperseg=nperseg
        )
        gpu_coherence = cp.asnumpy(gpu_coherence)

        assert array_equal(cpu_coherence, gpu_coherence)

    @pytest.mark.parametrize("num_samps", [2 ** 14])
    @pytest.mark.parametrize("fs", [1.0, 1e6])
    @pytest.mark.parametrize("nperseg", [1024, 2048])
    def test_coherence_complex(
        self, rand_complex_data_gen, num_samps, fs, nperseg
    ):
        cpu_x, gpu_x = rand_complex_data_gen(num_samps)
        cpu_y, gpu_y = rand_complex_data_gen(num_samps)

        _, cpu_coherence = signal.coherence(cpu_x, cpu_y, fs, nperseg=nperseg)
        _, gpu_coherence = cusignal.coherence(
            gpu_x, gpu_y, fs, nperseg=nperseg
        )
        gpu_coherence = cp.asnumpy(gpu_coherence)

        assert array_equal(cpu_coherence, gpu_coherence)

    @pytest.mark.parametrize("num_samps", [2 ** 14])
    @pytest.mark.parametrize("fs", [1.0, 1e6])
    @pytest.mark.parametrize("nperseg", [1024, 2048])
    def test_stft(self, rand_data_gen, num_samps, fs, nperseg):
        cpu_sig, gpu_sig = rand_data_gen(num_samps)

        _, _, cpu_stft = signal.stft(cpu_sig, fs, nperseg=nperseg)
        _, _, gpu_stft = cusignal.stft(gpu_sig, fs, nperseg=nperseg)
        gpu_stft = cp.asnumpy(gpu_stft)

        assert array_equal(cpu_stft, gpu_stft)

    @pytest.mark.parametrize("num_samps", [2 ** 14])
    @pytest.mark.parametrize("fs", [1.0, 1e6])
    @pytest.mark.parametrize("nperseg", [1024, 2048])
    def test_stft_complex(self, rand_complex_data_gen, num_samps, fs, nperseg):
        cpu_sig, gpu_sig = rand_complex_data_gen(num_samps)

        _, _, cpu_stft = signal.stft(cpu_sig, fs, nperseg=nperseg)
        _, _, gpu_stft = cusignal.stft(gpu_sig, fs, nperseg=nperseg)
        gpu_stft = cp.asnumpy(gpu_stft)

        assert array_equal(cpu_stft, gpu_stft)

    @pytest.mark.parametrize("num_in_samps", [2 ** 10])
    @pytest.mark.parametrize("num_out_samps", [2 ** 16, 2 ** 18])
    @pytest.mark.parametrize("precenter", [True, False])
    @pytest.mark.parametrize("normalize", [True, False])
    def test_lombscargle(
        self,
        lombscargle_gen,
        num_in_samps,
        num_out_samps,
        precenter,
        normalize,
    ):

        cpu_x, cpu_y, cpu_f, gpu_x, gpu_y, gpu_f = lombscargle_gen(
            num_in_samps, num_out_samps
        )

        cpu_lombscargle = signal.lombscargle(
            cpu_x, cpu_y, cpu_f, precenter, normalize
        )

        gpu_lombscargle = cp.asnumpy(
            cusignal.lombscargle(
                gpu_x, gpu_y, gpu_f, precenter, normalize,
            )
        )

        assert array_equal(cpu_lombscargle, gpu_lombscargle)
