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
from cusignal.test.utils import array_equal
import cusignal
import numpy as np
from scipy import signal


@pytest.mark.parametrize("num_samps", [2 ** 14])
@pytest.mark.parametrize("fs", [1.0, 1e6])
@pytest.mark.parametrize("nperseg", [1024, 2048])
def test_csd(num_samps, fs, nperseg):
    cpu_x = np.random.rand(num_samps)
    cpu_y = np.random.rand(num_samps)
    gpu_x = cp.asarray(cpu_x)
    gpu_y = cp.asarray(cpu_y)

    cpu_csd = signal.csd(cpu_x, cpu_y, fs, nperseg=nperseg)
    gpu_csd = cp.asnumpy(cusignal.csd(gpu_x, gpu_y, fs, nperseg=nperseg))

    assert array_equal(cpu_csd, gpu_csd)


@pytest.mark.parametrize("num_samps", [2 ** 14])
@pytest.mark.parametrize("fs", [1.0, 1e6])
@pytest.mark.parametrize("nperseg", [1024, 2048])
def test_csd_complex(num_samps, fs, nperseg):
    cpu_x = np.random.rand(num_samps) + 1j * np.random.rand(num_samps)
    cpu_y = np.random.rand(num_samps) + 1j * np.random.rand(num_samps)
    gpu_x = cp.asarray(cpu_x)
    gpu_y = cp.asarray(cpu_y)

    cpu_csd = signal.csd(cpu_x, cpu_y, fs, nperseg=nperseg)
    gpu_csd = cp.asnumpy(cusignal.csd(gpu_x, gpu_y, fs, nperseg=nperseg))

    assert array_equal(cpu_csd, gpu_csd)


@pytest.mark.parametrize("num_samps", [2 ** 14])
@pytest.mark.parametrize("fs", [1.0, 1e6])
@pytest.mark.parametrize("window", ["flattop", "nuttall"])
@pytest.mark.parametrize("scaling", ["spectrum", "density"])
def test_periodogram(num_samps, fs, window, scaling):
    cpu_sig = np.random.rand(num_samps)
    gpu_sig = cp.asarray(cpu_sig)

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
def test_periodogram_complex(num_samps, fs, window, scaling):
    cpu_sig = np.random.rand(num_samps) + 1j * np.random.rand(num_samps)
    gpu_sig = cp.asarray(cpu_sig)

    cf, cpu_periodogram = signal.periodogram(
        cpu_sig, fs, window=window, scaling=scaling
    )
    gf, gpu_periodogram = cusignal.periodogram(
        gpu_sig, fs, window=window, scaling=scaling
    )
    gpu_periodogram = cp.asnumpy(gpu_periodogram)

    assert array_equal(cpu_periodogram, gpu_periodogram)


@pytest.mark.parametrize("num_samps", [2 ** 14])
@pytest.mark.parametrize("fs", [1.0, 1e6])
@pytest.mark.parametrize("nperseg", [1024, 2048])
def test_welch(num_samps, fs, nperseg):
    cpu_sig = np.random.rand(num_samps)
    gpu_sig = cp.asarray(cpu_sig)

    cf, cPxx_spec = signal.welch(cpu_sig, fs, nperseg=nperseg)
    gf, gPxx_spec = cusignal.welch(gpu_sig, fs, nperseg=nperseg)
    gPxx_spec = cp.asnumpy(gPxx_spec)

    assert array_equal(cPxx_spec, gPxx_spec)


@pytest.mark.parametrize("num_samps", [2 ** 14])
@pytest.mark.parametrize("fs", [1.0, 1e6])
@pytest.mark.parametrize("nperseg", [1024, 2048])
def test_welch_complex(num_samps, fs, nperseg):
    cpu_sig = np.random.rand(num_samps) + 1j * np.random.rand(num_samps)
    gpu_sig = cp.asarray(cpu_sig)

    cf, cPxx_spec = signal.welch(cpu_sig, fs, nperseg=nperseg)
    gf, gPxx_spec = cusignal.welch(gpu_sig, fs, nperseg=nperseg)
    gPxx_spec = cp.asnumpy(gPxx_spec)

    assert array_equal(cPxx_spec, gPxx_spec)


@pytest.mark.parametrize("num_samps", [2 ** 14])
@pytest.mark.parametrize("fs", [1.0, 1e6])
def test_spectrogram(num_samps, fs):
    cpu_sig = np.random.rand(num_samps)
    gpu_sig = cp.asarray(cpu_sig)

    cf, ct, cPxx_spec = signal.spectrogram(cpu_sig, fs)
    gf, gt, gPxx_spec = cusignal.spectrogram(gpu_sig, fs)
    gPxx_spec = cp.asnumpy(gPxx_spec)

    assert array_equal(cPxx_spec, gPxx_spec)


@pytest.mark.parametrize("num_samps", [2 ** 14])
@pytest.mark.parametrize("fs", [1.0, 1e6])
def test_spectrogram_complex(num_samps, fs):
    cpu_sig = np.random.rand(num_samps) + 1j * np.random.rand(num_samps)
    gpu_sig = cp.asarray(cpu_sig)

    cf, ct, cPxx_spec = signal.spectrogram(cpu_sig, fs)
    gf, gt, gPxx_spec = cusignal.spectrogram(gpu_sig, fs)
    gPxx_spec = cp.asnumpy(gPxx_spec)

    assert array_equal(cPxx_spec, gPxx_spec)


@pytest.mark.parametrize("num_samps", [2 ** 14])
@pytest.mark.parametrize("fs", [1.0, 1e6])
@pytest.mark.parametrize("nperseg", [1024, 2048])
def test_coherence(num_samps, fs, nperseg):
    cpu_x = np.random.rand(num_samps)
    cpu_y = np.random.rand(num_samps)
    gpu_x = cp.asarray(cpu_x)
    gpu_y = cp.asarray(cpu_y)

    cf, cpu_coherence = signal.coherence(cpu_x, cpu_y, fs, nperseg=nperseg)
    gf, gpu_coherence = cusignal.coherence(gpu_x, gpu_y, fs, nperseg=nperseg)
    gpu_coherence = cp.asnumpy(gpu_coherence)

    assert array_equal(cpu_coherence, gpu_coherence)


@pytest.mark.parametrize("num_samps", [2 ** 14])
@pytest.mark.parametrize("fs", [1.0, 1e6])
@pytest.mark.parametrize("nperseg", [1024, 2048])
def test_coherence_complex(num_samps, fs, nperseg):
    cpu_x = np.random.rand(num_samps) + 1j * np.random.rand(num_samps)
    cpu_y = np.random.rand(num_samps) + 1j * np.random.rand(num_samps)
    gpu_x = cp.asarray(cpu_x)
    gpu_y = cp.asarray(cpu_y)

    cf, cpu_coherence = signal.coherence(cpu_x, cpu_y, fs, nperseg=nperseg)
    gf, gpu_coherence = cusignal.coherence(gpu_x, gpu_y, fs, nperseg=nperseg)
    gpu_coherence = cp.asnumpy(gpu_coherence)

    assert array_equal(cpu_coherence, gpu_coherence)


@pytest.mark.parametrize("num_samps", [2 ** 14])
@pytest.mark.parametrize("fs", [1.0, 1e6])
@pytest.mark.parametrize("nperseg", [1024, 2048])
def test_stft(num_samps, fs, nperseg):
    cpu_sig = np.random.rand(num_samps)
    gpu_sig = cp.asarray(cpu_sig)

    cf, ct, cpu_stft = signal.stft(cpu_sig, fs, nperseg=nperseg)
    gf, gt, gpu_stft = cusignal.stft(gpu_sig, fs, nperseg=nperseg)
    gpu_stft = cp.asnumpy(gpu_stft)

    assert array_equal(cpu_stft, gpu_stft)


@pytest.mark.parametrize("num_samps", [2 ** 14])
@pytest.mark.parametrize("fs", [1.0, 1e6])
@pytest.mark.parametrize("nperseg", [1024, 2048])
def test_stft_complex(num_samps, fs, nperseg):
    cpu_sig = np.random.rand(num_samps) + 1j * np.random.rand(num_samps)
    gpu_sig = cp.asarray(cpu_sig)

    cf, ct, cpu_stft = signal.stft(cpu_sig, fs, nperseg=nperseg)
    gf, gt, gpu_stft = cusignal.stft(gpu_sig, fs, nperseg=nperseg)
    gpu_stft = cp.asnumpy(gpu_stft)

    assert array_equal(cpu_stft, gpu_stft)


@pytest.mark.parametrize("num_in_samps", [2 ** 10])
@pytest.mark.parametrize("num_out_samps", [2 ** 16, 2 ** 18])
@pytest.mark.parametrize("precenter", ["True", "False"])
@pytest.mark.parametrize("normalize", ["True", "False"])
def test_lombscargle(num_in_samps, num_out_samps, precenter, normalize):
    A = 2.0
    w = 1.0
    phi = 0.5 * np.pi
    frac_points = 0.9  # Fraction of points to select

    r = np.random.rand(num_in_samps)
    x = np.linspace(0.01, 10 * np.pi, num_in_samps)
    x = x[r >= frac_points]

    y = A * np.sin(w * x + phi)

    f = np.linspace(0.01, 10, num_out_samps)

    cpu_lombscargle = signal.lombscargle(x, y, f, precenter, normalize)

    d_x = cp.asarray(x)
    d_y = cp.asarray(y)
    d_f = cp.asarray(f)

    gpu_lombscargle = cp.asnumpy(
        cusignal.lombscargle(d_x, d_y, d_f, precenter, normalize)
    )

    assert array_equal(cpu_lombscargle, gpu_lombscargle)
