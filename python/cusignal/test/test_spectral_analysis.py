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

from cusignal.test.utils import array_equal, _check_rapids_pytest_benchmark
from scipy import signal

cusignal.precompile_kernels()

gpubenchmark = _check_rapids_pytest_benchmark()


# Missing
# vectorstrength


class TestSpectral:
    @pytest.mark.benchmark(group="LombScargle")
    @pytest.mark.parametrize("num_in_samps", [2 ** 10])
    @pytest.mark.parametrize("num_out_samps", [2 ** 16, 2 ** 18])
    @pytest.mark.parametrize("precenter", [True, False])
    @pytest.mark.parametrize("normalize", [True, False])
    class TestLombScargle:
        def cpu_version(self, x, y, f, precenter, normalize):
            return signal.lombscargle(x, y, f, precenter, normalize)

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
            cpu_x, cpu_y, cpu_f, _, _, _ = lombscargle_gen(
                num_in_samps, num_out_samps
            )

            benchmark(
                self.cpu_version, cpu_x, cpu_y, cpu_f, precenter, normalize
            )

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
                cusignal.lombscargle,
                gpu_x,
                gpu_y,
                gpu_f,
                precenter,
                normalize,
            )

            key = self.cpu_version(cpu_x, cpu_y, cpu_f, precenter, normalize)
            assert array_equal(cp.asnumpy(output), key)

    @pytest.mark.benchmark(group="Periodogram")
    @pytest.mark.parametrize("num_samps", [2 ** 14])
    @pytest.mark.parametrize("fs", [1.0, 1e6])
    @pytest.mark.parametrize("window", ["flattop", "nuttall"])
    @pytest.mark.parametrize("scaling", ["spectrum", "density"])
    class TestPeriodogram:
        def cpu_version(self, cpu_sig, fs, window, scaling):
            return signal.periodogram(
                cpu_sig, fs, window=window, scaling=scaling
            )

        @pytest.mark.cpu
        def test_periodogram_cpu(
            self, rand_data_gen, benchmark, num_samps, fs, window, scaling
        ):
            cpu_sig, _ = rand_data_gen(num_samps)
            benchmark(self.cpu_version, cpu_sig, fs, window, scaling)

        def test_periodogram_gpu(
            self, rand_data_gen, gpubenchmark, num_samps, fs, window, scaling
        ):

            cpu_sig, gpu_sig = rand_data_gen(num_samps)
            _, output = gpubenchmark(
                cusignal.periodogram,
                gpu_sig,
                fs,
                window=window,
                scaling=scaling,
            )

            _, key = self.cpu_version(cpu_sig, fs, window, scaling)
            assert array_equal(cp.asnumpy(output), key)

    @pytest.mark.benchmark(group="PeriodogramComplex")
    @pytest.mark.parametrize("num_samps", [2 ** 14])
    @pytest.mark.parametrize("fs", [1.0, 1e6])
    @pytest.mark.parametrize("window", ["flattop", "nuttall"])
    @pytest.mark.parametrize("scaling", ["spectrum", "density"])
    class TestPeriodogramComplex:
        def cpu_version(self, cpu_sig, fs, window, scaling):
            return signal.periodogram(
                cpu_sig, fs, window=window, scaling=scaling
            )

        @pytest.mark.cpu
        def test_periodogram_complex_cpu(
            self,
            rand_complex_data_gen,
            benchmark,
            num_samps,
            fs,
            window,
            scaling,
        ):
            cpu_sig, _ = rand_complex_data_gen(num_samps)
            benchmark(self.cpu_version, cpu_sig, fs, window, scaling)

        def test_periodogram_complex_gpu(
            self,
            rand_complex_data_gen,
            gpubenchmark,
            num_samps,
            fs,
            window,
            scaling,
        ):

            cpu_sig, gpu_sig = rand_complex_data_gen(num_samps)
            _, output = gpubenchmark(
                cusignal.periodogram,
                gpu_sig,
                fs,
                window=window,
                scaling=scaling,
            )

            _, key = self.cpu_version(cpu_sig, fs, window, scaling)
            assert array_equal(cp.asnumpy(output), key)

    @pytest.mark.benchmark(group="Welch")
    @pytest.mark.parametrize("num_samps", [2 ** 14])
    @pytest.mark.parametrize("fs", [1.0, 1e6])
    @pytest.mark.parametrize("nperseg", [1024, 2048])
    class TestWelch:
        def cpu_version(self, cpu_sig, fs, nperseg):
            return signal.welch(cpu_sig, fs, nperseg=nperseg)

        @pytest.mark.cpu
        def test_welch_cpu(
            self, rand_data_gen, benchmark, num_samps, fs, nperseg
        ):
            cpu_sig, _ = rand_data_gen(num_samps)
            benchmark(self.cpu_version, cpu_sig, fs, nperseg)

        def test_welch_gpu(
            self, rand_data_gen, gpubenchmark, num_samps, fs, nperseg
        ):

            cpu_sig, gpu_sig = rand_data_gen(num_samps)
            _, output = gpubenchmark(
                cusignal.welch, gpu_sig, fs, nperseg=nperseg
            )

            _, key = self.cpu_version(cpu_sig, fs, nperseg)
            assert array_equal(cp.asnumpy(output), key)

    @pytest.mark.benchmark(group="WelchComplex")
    @pytest.mark.parametrize("num_samps", [2 ** 14])
    @pytest.mark.parametrize("fs", [1.0, 1e6])
    @pytest.mark.parametrize("nperseg", [1024, 2048])
    class TestWelchComplex:
        def cpu_version(self, cpu_sig, fs, nperseg):
            return signal.welch(cpu_sig, fs, nperseg=nperseg)

        @pytest.mark.cpu
        def test_welch_complex_cpu(
            self, rand_complex_data_gen, benchmark, num_samps, fs, nperseg
        ):
            cpu_sig, _ = rand_complex_data_gen(num_samps)
            benchmark(self.cpu_version, cpu_sig, fs, nperseg)

        def test_welch_complex_gpu(
            self, rand_complex_data_gen, gpubenchmark, num_samps, fs, nperseg
        ):

            cpu_sig, gpu_sig = rand_complex_data_gen(num_samps)
            _, output = gpubenchmark(
                cusignal.welch, gpu_sig, fs, nperseg=nperseg
            )

            _, key = self.cpu_version(cpu_sig, fs, nperseg)
            assert array_equal(cp.asnumpy(output), key)

    @pytest.mark.benchmark(group="CSD")
    @pytest.mark.parametrize("num_samps", [2 ** 14])
    @pytest.mark.parametrize("fs", [1.0, 1e6])
    @pytest.mark.parametrize("nperseg", [1024, 2048])
    class TestCSD:
        def cpu_version(self, cpu_x, cpu_y, fs, nperseg):
            return signal.csd(cpu_x, cpu_y, fs, nperseg=nperseg)

        @pytest.mark.cpu
        def test_csd_cpu(
            self, rand_data_gen, benchmark, num_samps, fs, nperseg
        ):
            cpu_x, _ = rand_data_gen(num_samps)
            cpu_y, _ = rand_data_gen(num_samps)
            benchmark(self.cpu_version, cpu_x, cpu_y, fs, nperseg)

        def test_csd_gpu(
            self, rand_data_gen, gpubenchmark, num_samps, fs, nperseg
        ):

            cpu_x, gpu_x = rand_data_gen(num_samps)
            cpu_y, gpu_y = rand_data_gen(num_samps)

            _, output = gpubenchmark(
                cusignal.csd, gpu_x, gpu_y, fs, nperseg=nperseg
            )

            _, key = self.cpu_version(cpu_x, cpu_y, fs, nperseg)
            assert array_equal(cp.asnumpy(output), key)

    @pytest.mark.benchmark(group="CSDComplex")
    @pytest.mark.parametrize("num_samps", [2 ** 14])
    @pytest.mark.parametrize("fs", [1.0, 1e6])
    @pytest.mark.parametrize("nperseg", [1024, 2048])
    class TestCSDComplex:
        def cpu_version(self, cpu_x, cpu_y, fs, nperseg):
            return signal.csd(cpu_x, cpu_y, fs, nperseg=nperseg)

        @pytest.mark.cpu
        def test_csd_complex_cpu(
            self, rand_complex_data_gen, benchmark, num_samps, fs, nperseg
        ):
            cpu_x, _ = rand_complex_data_gen(num_samps)
            cpu_y, _ = rand_complex_data_gen(num_samps)
            benchmark(self.cpu_version, cpu_x, cpu_y, fs, nperseg)

        def test_csd_complex_gpu(
            self, rand_complex_data_gen, gpubenchmark, num_samps, fs, nperseg
        ):

            cpu_x, gpu_x = rand_complex_data_gen(num_samps)
            cpu_y, gpu_y = rand_complex_data_gen(num_samps)
            _, output = gpubenchmark(
                cusignal.csd, gpu_x, gpu_y, fs, nperseg=nperseg
            )

            _, key = self.cpu_version(cpu_x, cpu_y, fs, nperseg)
            assert array_equal(cp.asnumpy(output), key)

    @pytest.mark.benchmark(group="Spectrogram")
    @pytest.mark.parametrize("num_samps", [2 ** 14])
    @pytest.mark.parametrize("fs", [1.0, 1e6])
    @pytest.mark.parametrize("nperseg", [1024, 2048])
    class TestSpectrogram:
        def cpu_version(self, cpu_sig, fs, nperseg):
            return signal.spectrogram(cpu_sig, fs, nperseg=nperseg)

        @pytest.mark.cpu
        def test_spectrogram_cpu(
            self, rand_data_gen, benchmark, num_samps, fs, nperseg
        ):
            cpu_sig, _ = rand_data_gen(num_samps)
            benchmark(self.cpu_version, cpu_sig, fs, nperseg)

        def test_spectrogram_gpu(
            self, rand_data_gen, gpubenchmark, num_samps, fs, nperseg
        ):

            cpu_sig, gpu_sig = rand_data_gen(num_samps)
            _, _, output = gpubenchmark(
                cusignal.spectrogram, gpu_sig, fs, nperseg=nperseg
            )

            _, _, key = self.cpu_version(cpu_sig, fs, nperseg)
            assert array_equal(cp.asnumpy(output), key)

    @pytest.mark.benchmark(group="SpectrogramComplex")
    @pytest.mark.parametrize("num_samps", [2 ** 14])
    @pytest.mark.parametrize("fs", [1.0, 1e6])
    @pytest.mark.parametrize("nperseg", [1024, 2048])
    class TestSpectrogramComplex:
        def cpu_version(self, cpu_sig, fs, nperseg):
            return signal.spectrogram(cpu_sig, fs, nperseg=nperseg)

        @pytest.mark.cpu
        def test_spectrogram_complex_cpu(
            self, rand_complex_data_gen, benchmark, num_samps, fs, nperseg
        ):
            cpu_sig, _ = rand_complex_data_gen(num_samps)
            benchmark(self.cpu_version, cpu_sig, fs, nperseg)

        def test_spectrogram_complex_gpu(
            self, rand_complex_data_gen, gpubenchmark, num_samps, fs, nperseg
        ):

            cpu_sig, gpu_sig = rand_complex_data_gen(num_samps)
            _, _, output = gpubenchmark(
                cusignal.spectrogram, gpu_sig, fs, nperseg=nperseg
            )

            _, _, key = self.cpu_version(cpu_sig, fs, nperseg)
            assert array_equal(cp.asnumpy(output), key)

    @pytest.mark.benchmark(group="STFT")
    @pytest.mark.parametrize("num_samps", [2 ** 14])
    @pytest.mark.parametrize("fs", [1.0, 1e6])
    @pytest.mark.parametrize("nperseg", [1024, 2048])
    class TestSTFT:
        def cpu_version(self, cpu_sig, fs, nperseg):
            return signal.stft(cpu_sig, fs, nperseg=nperseg)

        @pytest.mark.cpu
        def test_stft_cpu(
            self, rand_data_gen, benchmark, num_samps, fs, nperseg
        ):
            cpu_sig, _ = rand_data_gen(num_samps)
            benchmark(self.cpu_version, cpu_sig, fs, nperseg)

        def test_stft_gpu(
            self, rand_data_gen, gpubenchmark, num_samps, fs, nperseg
        ):

            cpu_sig, gpu_sig = rand_data_gen(num_samps)
            _, _, output = gpubenchmark(
                cusignal.stft, gpu_sig, fs, nperseg=nperseg
            )

            _, _, key = self.cpu_version(cpu_sig, fs, nperseg)
            assert array_equal(cp.asnumpy(output), key)

    @pytest.mark.benchmark(group="STFTComplex")
    @pytest.mark.parametrize("num_samps", [2 ** 14])
    @pytest.mark.parametrize("fs", [1.0, 1e6])
    @pytest.mark.parametrize("nperseg", [1024, 2048])
    class TestSTFTComplex:
        def cpu_version(self, cpu_sig, fs, nperseg):
            return signal.stft(cpu_sig, fs, nperseg=nperseg)

        @pytest.mark.cpu
        def test_stft_complex_cpu(
            self, rand_complex_data_gen, benchmark, num_samps, fs, nperseg
        ):
            cpu_sig, _ = rand_complex_data_gen(num_samps)
            benchmark(self.cpu_version, cpu_sig, fs, nperseg)

        def test_stft_complex_gpu(
            self, rand_complex_data_gen, gpubenchmark, num_samps, fs, nperseg
        ):

            cpu_sig, gpu_sig = rand_complex_data_gen(num_samps)
            _, _, output = gpubenchmark(
                cusignal.stft, gpu_sig, fs, nperseg=nperseg
            )

            _, _, key = self.cpu_version(cpu_sig, fs, nperseg)
            assert array_equal(cp.asnumpy(output), key)

    @pytest.mark.benchmark(group="Coherence")
    @pytest.mark.parametrize("num_samps", [2 ** 14])
    @pytest.mark.parametrize("fs", [1.0, 1e6])
    @pytest.mark.parametrize("nperseg", [1024, 2048])
    class TestCoherence:
        def cpu_version(self, cpu_x, cpu_y, fs, nperseg):
            return signal.coherence(cpu_x, cpu_y, fs, nperseg=nperseg)

        @pytest.mark.cpu
        def test_coherence_cpu(
            self, rand_data_gen, benchmark, num_samps, fs, nperseg
        ):
            cpu_x, _ = rand_data_gen(num_samps)
            cpu_y, _ = rand_data_gen(num_samps)
            benchmark(self.cpu_version, cpu_x, cpu_y, fs, nperseg)

        def test_coherence_gpu(
            self, rand_data_gen, gpubenchmark, num_samps, fs, nperseg
        ):
            cpu_x, gpu_x = rand_data_gen(num_samps)
            cpu_y, gpu_y = rand_data_gen(num_samps)

            _, output = gpubenchmark(
                cusignal.coherence, gpu_x, gpu_y, fs, nperseg=nperseg
            )

            _, key = self.cpu_version(cpu_x, cpu_y, fs, nperseg)
            assert array_equal(cp.asnumpy(output), key)

    @pytest.mark.benchmark(group="CoherenceComplex")
    @pytest.mark.parametrize("num_samps", [2 ** 14])
    @pytest.mark.parametrize("fs", [1.0, 1e6])
    @pytest.mark.parametrize("nperseg", [1024, 2048])
    class TestCoherenceComplex:
        def cpu_version(self, cpu_x, cpu_y, fs, nperseg):
            return signal.coherence(cpu_x, cpu_y, fs, nperseg=nperseg)

        @pytest.mark.cpu
        def test_coherence_complex_cpu(
            self, rand_complex_data_gen, benchmark, num_samps, fs, nperseg
        ):
            cpu_x, _ = rand_complex_data_gen(num_samps)
            cpu_y, _ = rand_complex_data_gen(num_samps)
            benchmark(self.cpu_version, cpu_x, cpu_y, fs, nperseg)

        def test_coherence_complex_gpu(
            self, rand_complex_data_gen, gpubenchmark, num_samps, fs, nperseg
        ):

            cpu_x, gpu_x = rand_complex_data_gen(num_samps)
            cpu_y, gpu_y = rand_complex_data_gen(num_samps)
            _, output = gpubenchmark(
                cusignal.coherence, gpu_x, gpu_y, fs, nperseg=nperseg
            )

            _, key = self.cpu_version(cpu_x, cpu_y, fs, nperseg)
            assert array_equal(cp.asnumpy(output), key)

    # @pytest.mark.benchmark(group="Vectorstrength")
    # class TestVectorstrength:
    #     def cpu_version(self, cpu_sig):
    #         return signal.vectorstrength(cpu_sig)

    #     @pytest.mark.cpu
    #     def test_vectorstrength_cpu(self, benchmark):
    #         benchmark(self.cpu_version, cpu_sig)

    #     def test_vectorstrength_gpu(self, gpubenchmark):

    #         output = gpubenchmark(cusignal.vectorstrength, gpu_sig)

    #         key = self.cpu_version(cpu_sig)
    #         assert array_equal(cp.asnumpy(output), key)
