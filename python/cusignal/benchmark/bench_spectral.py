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
from cusignal.test.utils import array_equal
import cusignal
from scipy import signal

cusignal.precompile_kernels()


@pytest.mark.benchmark(group="CSD")
@pytest.mark.parametrize("num_samps", [2 ** 14])
@pytest.mark.parametrize("fs", [1.0, 1e6])
@pytest.mark.parametrize("nperseg", [1024, 2048])
# Bench is required in class name for the test to be active
class BenchCSD:
    # This function will ensure the GPU version is getting the correct answer
    def cpu_version(self, cpu_x, cpu_y, fs, nperseg):
        return signal.csd(cpu_x, cpu_y, fs, nperseg=nperseg)

    # bench_ is required in function name to be searchable with -k parameter
    def bench_csd_cpu(self, rand_data_gen, benchmark, num_samps, fs, nperseg):
        cpu_x, _ = rand_data_gen(num_samps)
        cpu_y, _ = rand_data_gen(num_samps)
        benchmark(self.cpu_version, cpu_x, cpu_y, fs, nperseg)

    def bench_csd_gpu(self, rand_data_gen, benchmark, num_samps, fs, nperseg):

        cpu_x, gpu_x = rand_data_gen(num_samps)
        cpu_y, gpu_y = rand_data_gen(num_samps)
        # Variable output holds result from final cusignal.resample
        # It is not copied back until assert to so timing is not impacted
        _, output = benchmark(cusignal.csd, gpu_x, gpu_y, fs, nperseg=nperseg)

        _, key = self.cpu_version(cpu_x, cpu_y, fs, nperseg)
        assert array_equal(cp.asnumpy(output), key)


@pytest.mark.benchmark(group="CSDComplex")
@pytest.mark.parametrize("num_samps", [2 ** 14])
@pytest.mark.parametrize("fs", [1.0, 1e6])
@pytest.mark.parametrize("nperseg", [1024, 2048])
class BenchCSDComplex:
    def cpu_version(self, cpu_x, cpu_y, fs, nperseg):
        return signal.csd(cpu_x, cpu_y, fs, nperseg=nperseg)

    def bench_csd_complex_cpu(
        self, rand_complex_data_gen, benchmark, num_samps, fs, nperseg
    ):
        cpu_x, _ = rand_complex_data_gen(num_samps)
        cpu_y, _ = rand_complex_data_gen(num_samps)
        benchmark(self.cpu_version, cpu_x, cpu_y, fs, nperseg)

    def bench_csd_complex_gpu(
        self, rand_complex_data_gen, benchmark, num_samps, fs, nperseg
    ):

        cpu_x, gpu_x = rand_complex_data_gen(num_samps)
        cpu_y, gpu_y = rand_complex_data_gen(num_samps)
        _, output = benchmark(cusignal.csd, gpu_x, gpu_y, fs, nperseg=nperseg)

        _, key = self.cpu_version(cpu_x, cpu_y, fs, nperseg)
        assert array_equal(cp.asnumpy(output), key)


@pytest.mark.benchmark(group="Periodogram")
@pytest.mark.parametrize("num_samps", [2 ** 14])
@pytest.mark.parametrize("fs", [1.0, 1e6])
@pytest.mark.parametrize("window", ["flattop", "nuttall"])
@pytest.mark.parametrize("scaling", ["spectrum", "density"])
class BenchPeriodogram:
    def cpu_version(self, cpu_sig, fs, window, scaling):
        return signal.periodogram(cpu_sig, fs, window=window, scaling=scaling)

    def bench_periodogram_cpu(
        self, rand_data_gen, benchmark, num_samps, fs, window, scaling
    ):
        cpu_sig, _ = rand_data_gen(num_samps)
        benchmark(self.cpu_version, cpu_sig, fs, window, scaling)

    def bench_periodogram_gpu(
        self, rand_data_gen, benchmark, num_samps, fs, window, scaling
    ):

        cpu_sig, gpu_sig = rand_data_gen(num_samps)
        _, output = benchmark(
            cusignal.periodogram, gpu_sig, fs, window=window, scaling=scaling
        )

        _, key = self.cpu_version(cpu_sig, fs, window, scaling)
        assert array_equal(cp.asnumpy(output), key)


@pytest.mark.benchmark(group="PeriodogramComplex")
@pytest.mark.parametrize("num_samps", [2 ** 14])
@pytest.mark.parametrize("fs", [1.0, 1e6])
@pytest.mark.parametrize("window", ["flattop", "nuttall"])
@pytest.mark.parametrize("scaling", ["spectrum", "density"])
class BenchPeriodogramComplex:
    def cpu_version(self, cpu_sig, fs, window, scaling):
        return signal.periodogram(cpu_sig, fs, window=window, scaling=scaling)

    def bench_periodogram_complex_cpu(
        self, rand_complex_data_gen, benchmark, num_samps, fs, window, scaling
    ):
        cpu_sig, _ = rand_complex_data_gen(num_samps)
        benchmark(self.cpu_version, cpu_sig, fs, window, scaling)

    def bench_periodogram_complex_gpu(
        self, rand_complex_data_gen, benchmark, num_samps, fs, window, scaling
    ):

        cpu_sig, gpu_sig = rand_complex_data_gen(num_samps)
        _, output = benchmark(
            cusignal.periodogram, gpu_sig, fs, window=window, scaling=scaling
        )

        _, key = self.cpu_version(cpu_sig, fs, window, scaling)
        assert array_equal(cp.asnumpy(output), key)


@pytest.mark.benchmark(group="Welch")
@pytest.mark.parametrize("num_samps", [2 ** 14])
@pytest.mark.parametrize("fs", [1.0, 1e6])
@pytest.mark.parametrize("nperseg", [1024, 2048])
class BenchWelch:
    def cpu_version(self, cpu_sig, fs, nperseg):
        return signal.welch(cpu_sig, fs, nperseg=nperseg)

    def bench_welch_cpu(
        self, rand_data_gen, benchmark, num_samps, fs, nperseg
    ):
        cpu_sig, _ = rand_data_gen(num_samps)
        benchmark(self.cpu_version, cpu_sig, fs, nperseg)

    def bench_welch_gpu(
        self, rand_data_gen, benchmark, num_samps, fs, nperseg
    ):

        cpu_sig, gpu_sig = rand_data_gen(num_samps)
        _, output = benchmark(cusignal.welch, gpu_sig, fs, nperseg=nperseg)

        _, key = self.cpu_version(cpu_sig, fs, nperseg)
        assert array_equal(cp.asnumpy(output), key)


@pytest.mark.benchmark(group="WelchComplex")
@pytest.mark.parametrize("num_samps", [2 ** 14])
@pytest.mark.parametrize("fs", [1.0, 1e6])
@pytest.mark.parametrize("nperseg", [1024, 2048])
class BenchWelchComplex:
    def cpu_version(self, cpu_sig, fs, nperseg):
        return signal.welch(cpu_sig, fs, nperseg=nperseg)

    def bench_welch_complex_cpu(
        self, rand_complex_data_gen, benchmark, num_samps, fs, nperseg
    ):
        cpu_sig, _ = rand_complex_data_gen(num_samps)
        benchmark(self.cpu_version, cpu_sig, fs, nperseg)

    def bench_welch_complex_gpu(
        self, rand_complex_data_gen, benchmark, num_samps, fs, nperseg
    ):

        cpu_sig, gpu_sig = rand_complex_data_gen(num_samps)
        _, output = benchmark(cusignal.welch, gpu_sig, fs, nperseg=nperseg)

        _, key = self.cpu_version(cpu_sig, fs, nperseg)
        assert array_equal(cp.asnumpy(output), key)


@pytest.mark.benchmark(group="Spectrogram")
@pytest.mark.parametrize("num_samps", [2 ** 14])
@pytest.mark.parametrize("fs", [1.0, 1e6])
@pytest.mark.parametrize("nperseg", [1024, 2048])
class BenchSpectrogram:
    def cpu_version(self, cpu_sig, fs, nperseg):
        return signal.spectrogram(cpu_sig, fs, nperseg=nperseg)

    def bench_spectrogram_cpu(
        self, rand_data_gen, benchmark, num_samps, fs, nperseg
    ):
        cpu_sig, _ = rand_data_gen(num_samps)
        benchmark(self.cpu_version, cpu_sig, fs, nperseg)

    def bench_spectrogram_gpu(
        self, rand_data_gen, benchmark, num_samps, fs, nperseg
    ):

        cpu_sig, gpu_sig = rand_data_gen(num_samps)
        _, _, output = benchmark(
            cusignal.spectrogram, gpu_sig, fs, nperseg=nperseg
        )

        _, _, key = self.cpu_version(cpu_sig, fs, nperseg)
        assert array_equal(cp.asnumpy(output), key)


@pytest.mark.benchmark(group="SpectrogramComplex")
@pytest.mark.parametrize("num_samps", [2 ** 14])
@pytest.mark.parametrize("fs", [1.0, 1e6])
@pytest.mark.parametrize("nperseg", [1024, 2048])
class BenchSpectrogramComplex:
    def cpu_version(self, cpu_sig, fs, nperseg):
        return signal.spectrogram(cpu_sig, fs, nperseg=nperseg)

    def bench_spectrogram_complex_cpu(
        self, rand_complex_data_gen, benchmark, num_samps, fs, nperseg
    ):
        cpu_sig, _ = rand_complex_data_gen(num_samps)
        benchmark(self.cpu_version, cpu_sig, fs, nperseg)

    def bench_spectrogram_complex_gpu(
        self, rand_complex_data_gen, benchmark, num_samps, fs, nperseg
    ):

        cpu_sig, gpu_sig = rand_complex_data_gen(num_samps)
        _, _, output = benchmark(
            cusignal.spectrogram, gpu_sig, fs, nperseg=nperseg
        )

        _, _, key = self.cpu_version(cpu_sig, fs, nperseg)
        assert array_equal(cp.asnumpy(output), key)


@pytest.mark.benchmark(group="Coherence")
@pytest.mark.parametrize("num_samps", [2 ** 14])
@pytest.mark.parametrize("fs", [1.0, 1e6])
@pytest.mark.parametrize("nperseg", [1024, 2048])
class BenchCoherence:
    def cpu_version(self, cpu_x, cpu_y, fs, nperseg):
        return signal.coherence(cpu_x, cpu_y, fs, nperseg=nperseg)

    def bench_coherence_cpu(
        self, rand_data_gen, benchmark, num_samps, fs, nperseg
    ):
        cpu_x, _ = rand_data_gen(num_samps)
        cpu_y, _ = rand_data_gen(num_samps)
        benchmark(self.cpu_version, cpu_x, cpu_y, fs, nperseg)

    def bench_coherence_gpu(
        self, rand_data_gen, benchmark, num_samps, fs, nperseg
    ):

        cpu_x, gpu_x = rand_data_gen(num_samps)
        cpu_y, gpu_y = rand_data_gen(num_samps)
        _, output = benchmark(
            cusignal.coherence, gpu_x, gpu_y, fs, nperseg=nperseg
        )

        _, key = self.cpu_version(cpu_x, cpu_y, fs, nperseg)
        assert array_equal(cp.asnumpy(output), key)


@pytest.mark.benchmark(group="CoherenceComplex")
@pytest.mark.parametrize("num_samps", [2 ** 14])
@pytest.mark.parametrize("fs", [1.0, 1e6])
@pytest.mark.parametrize("nperseg", [1024, 2048])
class BenchCoherenceComplex:
    def cpu_version(self, cpu_x, cpu_y, fs, nperseg):
        return signal.coherence(cpu_x, cpu_y, fs, nperseg=nperseg)

    def bench_coherence_complex_cpu(
        self, rand_complex_data_gen, benchmark, num_samps, fs, nperseg
    ):
        cpu_x, _ = rand_complex_data_gen(num_samps)
        cpu_y, _ = rand_complex_data_gen(num_samps)
        benchmark(self.cpu_version, cpu_x, cpu_y, fs, nperseg)

    def bench_coherence_complex_gpu(
        self, rand_complex_data_gen, benchmark, num_samps, fs, nperseg
    ):

        cpu_x, gpu_x = rand_complex_data_gen(num_samps)
        cpu_y, gpu_y = rand_complex_data_gen(num_samps)
        _, output = benchmark(
            cusignal.coherence, gpu_x, gpu_y, fs, nperseg=nperseg
        )

        _, key = self.cpu_version(cpu_x, cpu_y, fs, nperseg)
        assert array_equal(cp.asnumpy(output), key)


@pytest.mark.benchmark(group="STFT")
@pytest.mark.parametrize("num_samps", [2 ** 14])
@pytest.mark.parametrize("fs", [1.0, 1e6])
@pytest.mark.parametrize("nperseg", [1024, 2048])
class BenchSTFT:
    def cpu_version(self, cpu_sig, fs, nperseg):
        return signal.stft(cpu_sig, fs, nperseg=nperseg)

    def bench_stft_cpu(self, rand_data_gen, benchmark, num_samps, fs, nperseg):
        cpu_sig, _ = rand_data_gen(num_samps)
        benchmark(self.cpu_version, cpu_sig, fs, nperseg)

    def bench_stft_gpu(self, rand_data_gen, benchmark, num_samps, fs, nperseg):

        cpu_sig, gpu_sig = rand_data_gen(num_samps)
        _, _, output = benchmark(cusignal.stft, gpu_sig, fs, nperseg=nperseg)

        _, _, key = self.cpu_version(cpu_sig, fs, nperseg)
        assert array_equal(cp.asnumpy(output), key)


@pytest.mark.benchmark(group="STFTComplex")
@pytest.mark.parametrize("num_samps", [2 ** 14])
@pytest.mark.parametrize("fs", [1.0, 1e6])
@pytest.mark.parametrize("nperseg", [1024, 2048])
class BenchSTFTComplex:
    def cpu_version(self, cpu_sig, fs, nperseg):
        return signal.stft(cpu_sig, fs, nperseg=nperseg)

    def bench_stft_complex_cpu(
        self, rand_complex_data_gen, benchmark, num_samps, fs, nperseg
    ):
        cpu_sig, _ = rand_complex_data_gen(num_samps)
        benchmark(self.cpu_version, cpu_sig, fs, nperseg)

    def bench_stft_complex_gpu(
        self, rand_complex_data_gen, benchmark, num_samps, fs, nperseg
    ):

        cpu_sig, gpu_sig = rand_complex_data_gen(num_samps)
        _, _, output = benchmark(cusignal.stft, gpu_sig, fs, nperseg=nperseg)

        _, _, key = self.cpu_version(cpu_sig, fs, nperseg)
        assert array_equal(cp.asnumpy(output), key)


@pytest.mark.benchmark(group="LombScargle")
@pytest.mark.parametrize("num_in_samps", [2 ** 10])
@pytest.mark.parametrize("num_out_samps", [2 ** 16, 2 ** 18])
@pytest.mark.parametrize("precenter", [True, False])
@pytest.mark.parametrize("normalize", [True, False])
class BenchLombScargle:
    def cpu_version(self, x, y, f, precenter, normalize):
        return signal.lombscargle(x, y, f, precenter, normalize)

    def bench_lombscargle_cpu(
        self,
        linspace_data_gen,
        rand_data_gen,
        benchmark,
        num_in_samps,
        num_out_samps,
        precenter,
        normalize,
    ):
        A = 2.0
        w = 1.0
        phi = 0.5 * np.pi
        frac_points = 0.9  # Fraction of points to select

        r, _ = rand_data_gen(num_in_samps)
        x, _ = linspace_data_gen(0.01, 10 * np.pi, num_in_samps)
        x = x[r >= frac_points]

        y = A * np.sin(w * x + phi)

        f, _ = linspace_data_gen(0.01, 10, num_out_samps)

        benchmark(self.cpu_version, x, y, f, precenter, normalize)

    def bench_lombscargle_gpu(
        self,
        linspace_data_gen,
        rand_data_gen,
        benchmark,
        num_in_samps,
        num_out_samps,
        precenter,
        normalize,
    ):
        A = 2.0
        w = 1.0
        phi = 0.5 * np.pi
        frac_points = 0.9  # Fraction of points to select

        r, _ = rand_data_gen(num_in_samps)
        x, _ = linspace_data_gen(0.01, 10 * np.pi, num_in_samps)
        x = x[r >= frac_points]
        y = A * np.sin(w * x + phi)
        f, d_f = linspace_data_gen(0.01, 10, num_out_samps)

        d_x = cp.asarray(x)
        d_y = cp.asarray(y)

        output = benchmark(
            cusignal.lombscargle,
            d_x,
            d_y,
            d_f,
            precenter,
            normalize,
        )

        key = self.cpu_version(x, y, f, precenter, normalize)
        assert array_equal(cp.asnumpy(output), key)
