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
from cusignal.testing.utils import _check_rapids_pytest_benchmark, array_equal

gpubenchmark = _check_rapids_pytest_benchmark()


class TestSpectral:
    @pytest.mark.benchmark(group="LombScargle")
    @pytest.mark.parametrize("num_in_samps", [2**10])
    @pytest.mark.parametrize("num_out_samps", [2**16, 2**18])
    @pytest.mark.parametrize("precenter", [True, False])
    @pytest.mark.parametrize("normalize", [True, False])
    class TestLombScargle:
        def cpu_version(self, x, y, f, precenter, normalize):
            return signal.lombscargle(x, y, f, precenter, normalize)

        def gpu_version(self, x, y, f, precenter, normalize):
            with cp.cuda.Stream.null:
                out = cusignal.lombscargle(x, y, f, precenter, normalize)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_lombscargle_cpu(
            self,
            lombscargle_gen,
            benchmark,
            num_in_samps,
            num_out_samps,
            precenter,
            normalize,
        ):
            cpu_x, cpu_y, cpu_f, _, _, _ = lombscargle_gen(num_in_samps, num_out_samps)

            benchmark(self.cpu_version, cpu_x, cpu_y, cpu_f, precenter, normalize)

        def test_lombscargle_gpu(
            self,
            lombscargle_gen,
            gpubenchmark,
            num_in_samps,
            num_out_samps,
            precenter,
            normalize,
        ):
            cpu_x, cpu_y, cpu_f, gpu_x, gpu_y, gpu_f = lombscargle_gen(
                num_in_samps, num_out_samps
            )
            output = gpubenchmark(
                self.gpu_version, gpu_x, gpu_y, gpu_f, precenter, normalize
            )

            key = self.cpu_version(cpu_x, cpu_y, cpu_f, precenter, normalize)
            array_equal(output, key)

    @pytest.mark.benchmark(group="Periodogram")
    @pytest.mark.parametrize("dtype", [np.float64, np.complex128])
    @pytest.mark.parametrize("num_samps", [2**14])
    @pytest.mark.parametrize("fs", [1.0, 1e6])
    @pytest.mark.parametrize("window", ["flattop", "nuttall"])
    @pytest.mark.parametrize("scaling", ["spectrum", "density"])
    class TestPeriodogram:
        def cpu_version(self, sig, fs, window, scaling):
            return signal.periodogram(sig, fs, window=window, scaling=scaling)

        def gpu_version(self, sig, fs, window, scaling):
            with cp.cuda.Stream.null:
                out = cusignal.periodogram(sig, fs, window=window, scaling=scaling)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_periodogram_cpu(
            self,
            rand_data_gen,
            benchmark,
            dtype,
            num_samps,
            fs,
            window,
            scaling,
        ):
            cpu_sig, _ = rand_data_gen(num_samps, 1, dtype)
            benchmark(self.cpu_version, cpu_sig, fs, window, scaling)

        def test_periodogram_gpu(
            self,
            rand_data_gen,
            gpubenchmark,
            dtype,
            num_samps,
            fs,
            window,
            scaling,
        ):

            cpu_sig, gpu_sig = rand_data_gen(num_samps, 1, dtype)
            output = gpubenchmark(self.gpu_version, gpu_sig, fs, window, scaling)

            key = self.cpu_version(cpu_sig, fs, window, scaling)
            array_equal(output, key)

    @pytest.mark.benchmark(group="Welch")
    @pytest.mark.parametrize("dtype", [np.float64, np.complex128])
    @pytest.mark.parametrize("num_samps", [2**14])
    @pytest.mark.parametrize("fs", [1.0, 1e6])
    @pytest.mark.parametrize("nperseg", [1024, 2048])
    class TestWelch:
        def cpu_version(self, sig, fs, nperseg):
            return signal.welch(sig, fs, nperseg=nperseg)

        def gpu_version(self, sig, fs, nperseg):
            with cp.cuda.Stream.null:
                out = cusignal.welch(sig, fs, nperseg=nperseg)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_welch_cpu(
            self, rand_data_gen, benchmark, dtype, num_samps, fs, nperseg
        ):
            cpu_sig, _ = rand_data_gen(num_samps, 1, dtype)
            benchmark(self.cpu_version, cpu_sig, fs, nperseg)

        def test_welch_gpu(
            self, rand_data_gen, gpubenchmark, dtype, num_samps, fs, nperseg
        ):

            cpu_sig, gpu_sig = rand_data_gen(num_samps, 1, dtype)
            output = gpubenchmark(self.gpu_version, cpu_sig, fs, nperseg)

            key = self.cpu_version(cpu_sig, fs, nperseg)
            array_equal(output, key)

    @pytest.mark.benchmark(group="CSD")
    @pytest.mark.parametrize("dtype", [np.float64, np.complex128])
    @pytest.mark.parametrize("num_samps", [2**14])
    @pytest.mark.parametrize("fs", [1.0, 1e6])
    @pytest.mark.parametrize("nperseg", [1024, 2048])
    class TestCSD:
        def cpu_version(self, x, y, fs, nperseg):
            return signal.csd(x, y, fs, nperseg=nperseg)

        def gpu_version(self, x, y, fs, nperseg):
            with cp.cuda.Stream.null:
                out = cusignal.csd(x, y, fs, nperseg=nperseg)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_csd_cpu(self, rand_data_gen, benchmark, dtype, num_samps, fs, nperseg):
            cpu_x, _ = rand_data_gen(num_samps, 1, dtype)
            cpu_y, _ = rand_data_gen(num_samps, 1, dtype)
            benchmark(self.cpu_version, cpu_x, cpu_y, fs, nperseg)

        def test_csd_gpu(
            self, rand_data_gen, gpubenchmark, dtype, num_samps, fs, nperseg
        ):

            cpu_x, gpu_x = rand_data_gen(num_samps, 1, dtype)
            cpu_y, gpu_y = rand_data_gen(num_samps, 1, dtype)

            output = gpubenchmark(self.gpu_version, gpu_x, gpu_y, fs, nperseg)

            key = self.cpu_version(cpu_x, cpu_y, fs, nperseg)
            array_equal(output, key)

    @pytest.mark.benchmark(group="Spectrogram")
    @pytest.mark.parametrize("dtype", [np.float64, np.complex128])
    @pytest.mark.parametrize("num_samps", [2**14])
    @pytest.mark.parametrize("fs", [1.0, 1e6])
    @pytest.mark.parametrize("nperseg", [1024, 2048])
    class TestSpectrogram:
        def cpu_version(self, sig, fs, nperseg):
            return signal.spectrogram(sig, fs, nperseg=nperseg)

        def gpu_version(self, sig, fs, nperseg):
            with cp.cuda.Stream.null:
                out = cusignal.spectrogram(sig, fs, nperseg=nperseg)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_spectrogram_cpu(
            self, rand_data_gen, benchmark, dtype, num_samps, fs, nperseg
        ):
            cpu_sig, _ = rand_data_gen(num_samps, 1, dtype)
            benchmark(self.cpu_version, cpu_sig, fs, nperseg)

        def test_spectrogram_gpu(
            self, rand_data_gen, gpubenchmark, dtype, num_samps, fs, nperseg
        ):

            cpu_sig, gpu_sig = rand_data_gen(num_samps, 1, dtype)
            output = gpubenchmark(self.gpu_version, gpu_sig, fs, nperseg)

            key = self.cpu_version(cpu_sig, fs, nperseg)
            array_equal(output, key)

    @pytest.mark.benchmark(group="STFT")
    @pytest.mark.parametrize("dtype", [np.float64, np.complex128])
    @pytest.mark.parametrize("num_samps", [2**14])
    @pytest.mark.parametrize("fs", [1.0, 1e6])
    @pytest.mark.parametrize("nperseg", [1024, 2048])
    class TestSTFT:
        def cpu_version(self, sig, fs, nperseg):
            return signal.stft(sig, fs, nperseg=nperseg)

        def gpu_version(self, sig, fs, nperseg):
            with cp.cuda.Stream.null:
                out = cusignal.stft(sig, fs, nperseg=nperseg)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_stft_cpu(
            self, rand_data_gen, benchmark, dtype, num_samps, fs, nperseg
        ):
            cpu_sig, _ = rand_data_gen(num_samps, 1, dtype)
            benchmark(self.cpu_version, cpu_sig, fs, nperseg)

        def test_stft_gpu(
            self, rand_data_gen, gpubenchmark, dtype, num_samps, fs, nperseg
        ):

            cpu_sig, gpu_sig = rand_data_gen(num_samps, 1, dtype)
            output = gpubenchmark(self.gpu_version, gpu_sig, fs, nperseg)

            key = self.cpu_version(cpu_sig, fs, nperseg)
            array_equal(output, key)

    @pytest.mark.benchmark(group="ISTFT")
    @pytest.mark.parametrize("dtype", [np.float64, np.complex128])
    @pytest.mark.parametrize("num_samps", [2**16])
    @pytest.mark.parametrize("fs", [1.0, 1e6])
    @pytest.mark.parametrize("nperseg", [1024, 2048])
    class TestISTFT:
        def cpu_version(self, sig, fs, nperseg):
            return signal.istft(sig, fs, nperseg=nperseg)

        def gpu_version(self, sig, fs, nperseg):
            with cp.cuda.Stream.null:
                out = cusignal.istft(sig, fs, nperseg=nperseg)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_istft_cpu(
            self, rand_data_gen, benchmark, dtype, num_samps, fs, nperseg
        ):
            cpu_sig, _ = rand_data_gen(num_samps, 1, dtype)
            _, _, cpu_sig = signal.stft(cpu_sig, fs, nperseg=nperseg)
            benchmark(self.cpu_version, cpu_sig, fs, nperseg)

        def test_istft_gpu(
            self, rand_data_gen, gpubenchmark, dtype, num_samps, fs, nperseg
        ):

            cpu_sig, gpu_sig = rand_data_gen(num_samps, 1, dtype)
            _, _, cpu_sig = signal.stft(cpu_sig, fs, nperseg=nperseg)
            _, _, gpu_sig = cusignal.stft(gpu_sig, fs, nperseg=nperseg)
            output = gpubenchmark(self.gpu_version, gpu_sig, fs, nperseg)

            key = self.cpu_version(cpu_sig, fs, nperseg)
            array_equal(output, key)

    @pytest.mark.benchmark(group="Coherence")
    @pytest.mark.parametrize("dtype", [np.float64, np.complex128])
    @pytest.mark.parametrize("num_samps", [2**14])
    @pytest.mark.parametrize("fs", [1.0, 1e6])
    @pytest.mark.parametrize("nperseg", [1024, 2048])
    class TestCoherence:
        def cpu_version(self, x, y, fs, nperseg):
            return signal.coherence(x, y, fs, nperseg=nperseg)

        def gpu_version(self, x, y, fs, nperseg):
            with cp.cuda.Stream.null:
                out = cusignal.coherence(x, y, fs, nperseg=nperseg)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_coherence_cpu(
            self, rand_data_gen, benchmark, dtype, num_samps, fs, nperseg
        ):
            cpu_x, _ = rand_data_gen(num_samps, 1, dtype)
            cpu_y, _ = rand_data_gen(num_samps, 1, dtype)
            benchmark(self.cpu_version, cpu_x, cpu_y, fs, nperseg)

        def test_coherence_gpu(
            self, rand_data_gen, gpubenchmark, dtype, num_samps, fs, nperseg
        ):
            cpu_x, gpu_x = rand_data_gen(num_samps, 1, dtype)
            cpu_y, gpu_y = rand_data_gen(num_samps, 1, dtype)

            output = gpubenchmark(self.gpu_version, gpu_x, gpu_y, fs, nperseg)

            key = self.cpu_version(cpu_x, cpu_y, fs, nperseg)
            array_equal(output, key)

    @pytest.mark.benchmark(group="Vectorstrength")
    @pytest.mark.parametrize("period", [0.75, 5])
    @pytest.mark.parametrize("num_samps", [2**4])
    class TestVectorstrength:
        def cpu_version(self, events, period):
            return signal.vectorstrength(events, period)

        def gpu_version(self, events, period):
            with cp.cuda.Stream.null:
                out = cusignal.vectorstrength(events, period)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_vectorstrength_cpu(self, time_data_gen, benchmark, num_samps, period):
            events_cpu, _ = time_data_gen(0, 10, num_samps)
            benchmark(self.cpu_version, events_cpu, period)

        def test_vectorstrength_gpu(
            self, time_data_gen, gpubenchmark, num_samps, period
        ):
            events_cpu, events_gpu = time_data_gen(0, 10, num_samps)
            period_gpu = cp.asarray(period)

            output = gpubenchmark(self.gpu_version, events_gpu, period_gpu)

            key = self.cpu_version(events_cpu, period)
            array_equal(output, key)
