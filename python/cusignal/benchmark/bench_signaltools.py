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
import pytest-benchmark
import cupy as cp
import numpy as np
from cusignal.test.utils import array_equal
import cusignal
from scipy import signal


@pytest.mark.benchmark(group="Resample")
@pytest.mark.parametrize("num_samps", [2 ** 14])
@pytest.mark.parametrize("resample_num_samps", [2 ** 12, 2 ** 16])
@pytest.mark.parametrize("window", [("kaiser", 0.5)])
# Bench is required in class name for the test to be active
class BenchResample:
    # This function will ensure the GPU version is getting the correct answer
    def cpu_version(self, cpu_sig, resample_num_samps, window):
        return signal.resample(cpu_sig, resample_num_samps, window=window)

    # bench_ is required in function name to be searchable with -k parameter
    def bench_resample_cpu(
        self,
        resample_data_gen,
        benchmark,
        num_samps,
        resample_num_samps,
        window,
    ):
        cpu_sig, _ = resample_data_gen(0, 10, num_samps, endpoint=False)
        benchmark(
            self.cpu_version, cpu_sig, resample_num_samps, window,
        )

    def bench_resample_gpu(
        self,
        resample_data_gen,
        benchmark,
        num_samps,
        resample_num_samps,
        window,
    ):

        cpu_sig, gpu_sig = resample_data_gen(0, 10, num_samps, endpoint=False)
        # Variable output holds result from final cusignal.resample
        # It is not copied back until assert to so timing is not impacted
        output = benchmark(
            cusignal.resample, gpu_sig, resample_num_samps, window=window
        )

        key = self.cpu_version(cpu_sig, resample_num_samps, window)
        assert array_equal(cp.asnumpy(output), key)


@pytest.mark.benchmark(group="ResamplePoly")
@pytest.mark.parametrize("num_samps", [2 ** 14])
@pytest.mark.parametrize("up", [2, 3, 7])
@pytest.mark.parametrize("down", [1, 2, 9])
@pytest.mark.parametrize("window", [("kaiser", 0.5)])
class BenchResamplePoly:
    # This function will ensure the GPU version is getting the correct answer
    def cpu_version(self, cpu_sig, up, down, window):
        return signal.resample_poly(cpu_sig, up, down, window=window)

    def bench_resample_poly_cpu(
        self, resample_data_gen, benchmark, num_samps, up, down, window
    ):
        cpu_sig, _ = resample_data_gen(0, 10, num_samps, endpoint=False)
        benchmark(
            self.cpu_version, cpu_sig, up, down, window,
        )

    # Add parameter 'use_numba' to GPU test. It is not needed on CPU test.
    # This reduces redundant executions and runtime
    @pytest.mark.parametrize("use_numba", [True, False])
    def bench_resample_poly_gpu(
        self,
        resample_data_gen,
        benchmark,
        num_samps,
        up,
        down,
        window,
        use_numba,
    ):

        cpu_sig, gpu_sig = resample_data_gen(0, 10, num_samps, endpoint=False)
        # Variable output holds result from final cusignal.resample
        # It is not copied back until assert to so timing is not impacted
        output = benchmark(
            cusignal.resample_poly,
            gpu_sig,
            up,
            down,
            window=window,
            use_numba=use_numba,
        )

        key = self.cpu_version(cpu_sig, up, down, window)
        assert array_equal(cp.asnumpy(output), key)


@pytest.mark.benchmark(group="FirWin")
@pytest.mark.parametrize("num_samps", [2 ** 15])
@pytest.mark.parametrize("f1", [0.1, 0.15])
@pytest.mark.parametrize("f2", [0.2, 0.4])
class BenchFirWin:
    def cpu_version(self, num_samps, f1, f2):
        return signal.firwin(num_samps, [f1, f2], pass_zero=False)

    def bench_firwin_cpu(self, benchmark, num_samps, f1, f2):
        benchmark(
            self.cpu_version, num_samps, f1, f2,
        )

    def bench_firwin_gpu(self, benchmark, num_samps, f1, f2):

        output = benchmark(
            cusignal.firwin, num_samps, [f1, f2], pass_zero=False
        )

        key = self.cpu_version(num_samps, f1, f2)
        assert array_equal(cp.asnumpy(output), key)


@pytest.mark.benchmark(group="Correlate")
@pytest.mark.parametrize("num_samps", [2 ** 15])
@pytest.mark.parametrize("num_taps", [128, 2 ** 8, 2 ** 15])
@pytest.mark.parametrize("mode", ["full", "valid", "same"])
class BenchCorrelate:
    def cpu_version(self, cpu_sig, num_taps, mode):
        return signal.correlate(cpu_sig, num_taps, mode=mode)

    def bench_correlate_cpu(
        self, rand_data_gen, benchmark, num_samps, num_taps, mode
    ):
        cpu_sig, _ = rand_data_gen(num_samps)
        benchmark(self.cpu_version, cpu_sig, np.ones(num_taps), mode)

    def bench_correlate_gpu(
        self, rand_data_gen, benchmark, num_samps, num_taps, mode
    ):

        cpu_sig, gpu_sig = rand_data_gen(num_samps)
        output = benchmark(
            cusignal.correlate, gpu_sig, cp.ones(num_taps), mode=mode
        )

        key = self.cpu_version(cpu_sig, np.ones(num_taps), mode)
        assert array_equal(cp.asnumpy(output), key)


@pytest.mark.benchmark(group="Convolve")
@pytest.mark.parametrize("num_samps", [2 ** 15])
@pytest.mark.parametrize("num_taps", [128, 2 ** 8, 2 ** 15])
@pytest.mark.parametrize("mode", ["full", "valid", "same"])
class BenchConvolve:
    def cpu_version(self, cpu_sig, cpu_win, mode):
        return signal.convolve(cpu_sig, cpu_win, mode=mode)

    def bench_convolve_cpu(
        self, rand_data_gen, benchmark, num_samps, num_taps, mode
    ):
        cpu_sig, _ = rand_data_gen(num_samps)
        cpu_win = signal.windows.hann(num_taps)

        benchmark(self.cpu_version, cpu_sig, cpu_win, mode)

    def bench_convolve_gpu(
        self, rand_data_gen, benchmark, num_samps, num_taps, mode
    ):

        cpu_sig, gpu_sig = rand_data_gen(num_samps)
        gpu_win = cusignal.windows.hann(num_taps)
        output = benchmark(cusignal.convolve, gpu_sig, gpu_win, mode=mode)

        cpu_win = signal.windows.hann(num_taps)
        key = self.cpu_version(cpu_sig, cpu_win, mode)
        assert array_equal(cp.asnumpy(output), key)


@pytest.mark.benchmark(group="FFTConvolve")
@pytest.mark.parametrize("num_samps", [2 ** 15])
@pytest.mark.parametrize("mode", ["full", "valid", "same"])
class BenchFFTConvolve:
    def cpu_version(self, cpu_sig, mode):
        return signal.fftconvolve(cpu_sig, cpu_sig[::-1], mode=mode)

    def bench_fftconvolve_cpu(self, rand_data_gen, benchmark, num_samps, mode):
        cpu_sig, _ = rand_data_gen(num_samps)
        benchmark(self.cpu_version, cpu_sig, mode)

    def bench_fftconvolve_gpu(self, rand_data_gen, benchmark, num_samps, mode):

        cpu_sig, gpu_sig = rand_data_gen(num_samps)
        output = benchmark(
            cusignal.fftconvolve, gpu_sig, gpu_sig[::-1], mode=mode
        )

        key = self.cpu_version(cpu_sig, mode)
        assert array_equal(cp.asnumpy(output), key)


@pytest.mark.benchmark(group="Wiener")
@pytest.mark.parametrize("num_samps", [2 ** 15, 2 ** 24])
class BenchWiener:
    def cpu_version(self, cpu_sig):
        return signal.wiener(cpu_sig)

    def bench_wiener_cpu(self, rand_data_gen, benchmark, num_samps):
        cpu_sig, _ = rand_data_gen(num_samps)
        benchmark(self.cpu_version, cpu_sig)

    def bench_wiener_gpu(self, rand_data_gen, benchmark, num_samps):

        cpu_sig, gpu_sig = rand_data_gen(num_samps)
        output = benchmark(cusignal.wiener, gpu_sig)

        key = self.cpu_version(cpu_sig)
        assert array_equal(cp.asnumpy(output), key)


@pytest.mark.benchmark(group="Hilbert")
@pytest.mark.parametrize("num_samps", [2 ** 15])
class BenchHilbert:
    def cpu_version(self, cpu_sig):
        return signal.hilbert(cpu_sig)

    def bench_hilbert_cpu(self, rand_data_gen, benchmark, num_samps):
        cpu_sig, _ = rand_data_gen(num_samps)
        benchmark(self.cpu_version, cpu_sig)

    def bench_hilbert_gpu(self, rand_data_gen, benchmark, num_samps):

        cpu_sig, gpu_sig = rand_data_gen(num_samps)
        output = benchmark(cusignal.hilbert, gpu_sig)

        key = self.cpu_version(cpu_sig)
        assert array_equal(cp.asnumpy(output), key)


@pytest.mark.benchmark(group="Hilbert2")
@pytest.mark.parametrize("num_samps", [2 ** 8])
class BenchHilbert2:
    def cpu_version(self, cpu_sig):
        return signal.hilbert2(cpu_sig)

    def bench_hilbert2_cpu(self, rand_2d_data_gen, benchmark, num_samps):
        cpu_sig, _ = rand_2d_data_gen(num_samps)
        benchmark(self.cpu_version, cpu_sig)

    def bench_hilbert2_gpu(self, rand_2d_data_gen, benchmark, num_samps):

        cpu_sig, gpu_sig = rand_2d_data_gen(num_samps)
        output = benchmark(cusignal.hilbert2, gpu_sig)

        key = self.cpu_version(cpu_sig)
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
        self, rand_2d_data_gen, benchmark, num_samps, num_taps, boundary, mode
    ):
        cpu_sig, _ = rand_2d_data_gen(num_samps)
        cpu_filt, _ = rand_2d_data_gen(num_taps)
        benchmark(self.cpu_version, cpu_sig, cpu_filt, boundary, mode)

    def bench_convolve2d_gpu(
        self, rand_2d_data_gen, benchmark, num_samps, num_taps, boundary, mode
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
        self, rand_2d_data_gen, benchmark, num_samps, num_taps, boundary, mode
    ):
        cpu_sig, _ = rand_2d_data_gen(num_samps)
        cpu_filt, _ = rand_2d_data_gen(num_taps)
        benchmark(self.cpu_version, cpu_sig, cpu_filt, boundary, mode)

    def bench_correlate2d_gpu(
        self, rand_2d_data_gen, benchmark, num_samps, num_taps, boundary, mode
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
