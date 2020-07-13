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

cusignal.precompile_kernels()


class BenchConvolution:
    @pytest.mark.benchmark(group="Correlate")
    @pytest.mark.parametrize("num_samps", [2 ** 7, 2 ** 10 + 1, 2 ** 13])
    @pytest.mark.parametrize("num_taps", [125, 2 ** 8, 2 ** 13])
    @pytest.mark.parametrize("mode", ["full", "valid", "same"])
    @pytest.mark.parametrize("method", ["direct", "fft", "auto"])
    class BenchCorrelate:
        def cpu_version(self, cpu_sig, num_taps, mode, method):
            return signal.correlate(
                cpu_sig, num_taps, mode=mode, method=method
            )

        def bench_correlate1d_cpu(
            self, rand_data_gen, benchmark, num_samps, num_taps, mode, method
        ):
            cpu_sig, _ = rand_data_gen(num_samps)
            benchmark(
                self.cpu_version, cpu_sig, np.ones(num_taps), mode, method
            )

        def bench_correlate1d_gpu(
            self, rand_data_gen, benchmark, num_samps, num_taps, mode, method
        ):

            cpu_sig, gpu_sig = rand_data_gen(num_samps)
            output = benchmark(
                cusignal.correlate,
                gpu_sig,
                cp.ones(num_taps),
                mode=mode,
                method=method,
            )

            key = self.cpu_version(cpu_sig, np.ones(num_taps), mode, method)
            assert array_equal(cp.asnumpy(output), key)

    @pytest.mark.benchmark(group="Convolve")
    @pytest.mark.parametrize("num_samps", [2 ** 7, 2 ** 10 + 1, 2 ** 13])
    @pytest.mark.parametrize("num_taps", [125, 2 ** 8, 2 ** 13])
    @pytest.mark.parametrize("mode", ["full", "valid", "same"])
    @pytest.mark.parametrize("method", ["direct", "fft", "auto"])
    class BenchConvolve:
        def cpu_version(self, cpu_sig, cpu_win, mode, method):
            return signal.convolve(cpu_sig, cpu_win, mode=mode, method=method)

        def bench_convolve1d_cpu(
            self, rand_data_gen, benchmark, num_samps, num_taps, mode, method
        ):
            cpu_sig, _ = rand_data_gen(num_samps)
            cpu_win = signal.windows.hann(num_taps)

            benchmark(self.cpu_version, cpu_sig, cpu_win, mode, method)

        def bench_convolve1d_gpu(
            self, rand_data_gen, benchmark, num_samps, num_taps, mode, method
        ):

            cpu_sig, gpu_sig = rand_data_gen(num_samps)
            gpu_win = cusignal.windows.hann(num_taps)
            output = benchmark(
                cusignal.convolve, gpu_sig, gpu_win, mode=mode, method=method
            )

            cpu_win = signal.windows.hann(num_taps)
            key = self.cpu_version(cpu_sig, cpu_win, mode, method)
            assert array_equal(cp.asnumpy(output), key)

    @pytest.mark.benchmark(group="FFTConvolve")
    @pytest.mark.parametrize("num_samps", [2 ** 15])
    @pytest.mark.parametrize("mode", ["full", "valid", "same"])
    class BenchFFTConvolve:
        def cpu_version(self, cpu_sig, mode):
            return signal.fftconvolve(cpu_sig, cpu_sig[::-1], mode=mode)

        def bench_fftconvolve_cpu(
            self, rand_data_gen, benchmark, num_samps, mode
        ):
            cpu_sig, _ = rand_data_gen(num_samps)
            benchmark(self.cpu_version, cpu_sig, mode)

        def bench_fftconvolve_gpu(
            self, rand_data_gen, benchmark, num_samps, mode
        ):

            cpu_sig, gpu_sig = rand_data_gen(num_samps)
            output = benchmark(
                cusignal.fftconvolve, gpu_sig, gpu_sig[::-1], mode=mode
            )

            key = self.cpu_version(cpu_sig, mode)
            assert array_equal(cp.asnumpy(output), key)

    @pytest.mark.benchmark(group="Convolve2d")
    @pytest.mark.parametrize("num_samps", [2 ** 8])
    @pytest.mark.parametrize("num_taps", [5, 100])
    @pytest.mark.parametrize("boundary", ["fill", "wrap", "symm"])
    @pytest.mark.parametrize("mode", ["full", "valid", "same"])
    class BenchConvolve2d:
        def cpu_version(self, cpu_sig, cpu_filt, boundary, mode):
            return signal.convolve2d(
                cpu_sig, cpu_filt, boundary=boundary, mode=mode
            )

        def bench_convolve2d_cpu(
            self,
            rand_2d_data_gen,
            benchmark,
            num_samps,
            num_taps,
            boundary,
            mode,
        ):
            cpu_sig, _ = rand_2d_data_gen(num_samps)
            cpu_filt, _ = rand_2d_data_gen(num_taps)
            benchmark(self.cpu_version, cpu_sig, cpu_filt, boundary, mode)

        def bench_convolve2d_gpu(
            self,
            rand_2d_data_gen,
            benchmark,
            num_samps,
            num_taps,
            boundary,
            mode,
        ):

            cpu_sig, gpu_sig = rand_2d_data_gen(num_samps)
            cpu_filt, gpu_filt = rand_2d_data_gen(num_taps)
            output = benchmark(
                cusignal.convolve2d,
                gpu_sig,
                gpu_filt,
                boundary=boundary,
                mode=mode,
            )

            key = self.cpu_version(cpu_sig, cpu_filt, boundary, mode)
            assert array_equal(cp.asnumpy(output), key)

    @pytest.mark.benchmark(group="Correlate2d")
    @pytest.mark.parametrize("num_samps", [2 ** 8])
    @pytest.mark.parametrize("num_taps", [5, 100])
    @pytest.mark.parametrize("boundary", ["fill", "wrap", "symm"])
    @pytest.mark.parametrize("mode", ["full", "valid", "same"])
    class BenchCorrelate2d:
        def cpu_version(self, cpu_sig, cpu_filt, boundary, mode):
            return signal.correlate2d(
                cpu_sig, cpu_filt, boundary=boundary, mode=mode
            )

        def bench_correlate2d_cpu(
            self,
            rand_2d_data_gen,
            benchmark,
            num_samps,
            num_taps,
            boundary,
            mode,
        ):
            cpu_sig, _ = rand_2d_data_gen(num_samps)
            cpu_filt, _ = rand_2d_data_gen(num_taps)
            benchmark(self.cpu_version, cpu_sig, cpu_filt, boundary, mode)

        def bench_correlate2d_gpu(
            self,
            rand_2d_data_gen,
            benchmark,
            num_samps,
            num_taps,
            boundary,
            mode,
        ):

            cpu_sig, gpu_sig = rand_2d_data_gen(num_samps)
            cpu_filt, gpu_filt = rand_2d_data_gen(num_taps)
            output = benchmark(
                cusignal.correlate2d,
                gpu_sig,
                gpu_filt,
                boundary=boundary,
                mode=mode,
            )

            key = self.cpu_version(cpu_sig, cpu_filt, boundary, mode)
            assert array_equal(cp.asnumpy(output), key)
