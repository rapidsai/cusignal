# Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import cupy as cp
import pytest
from scipy import signal

import cusignal
from cusignal.testing.utils import _check_rapids_pytest_benchmark, array_equal

gpubenchmark = _check_rapids_pytest_benchmark()


class TestConvolution:
    @pytest.mark.benchmark(group="Correlate")
    @pytest.mark.parametrize("num_samps", [2**7, 2**10 + 1, 2**13])
    @pytest.mark.parametrize("num_taps", [125, 2**8, 2**13])
    @pytest.mark.parametrize("mode", ["full", "valid", "same"])
    @pytest.mark.parametrize("method", ["direct", "fft", "auto"])
    class TestCorrelate:
        def cpu_version(self, sig, num_taps, mode, method):
            return signal.correlate(sig, num_taps, mode=mode, method=method)

        def gpu_version(self, sig, num_taps, mode, method):
            with cp.cuda.Stream.null:
                out = cusignal.correlate(sig, num_taps, mode=mode, method=method)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_correlate1d_cpu(
            self,
            rand_data_gen,
            benchmark,
            num_samps,
            num_taps,
            mode,
            method,
        ):
            cpu_sig, _ = rand_data_gen(num_samps, 1)
            cpu_filt, _ = rand_data_gen(num_taps, 1)
            benchmark(self.cpu_version, cpu_sig, cpu_filt, mode, method)

        def test_correlate1d_gpu(
            self,
            rand_data_gen,
            gpubenchmark,
            num_samps,
            num_taps,
            mode,
            method,
        ):

            cpu_sig, gpu_sig = rand_data_gen(num_samps, 1)
            cpu_filt, gpu_filt = rand_data_gen(num_taps, 1)
            output = gpubenchmark(
                self.gpu_version,
                gpu_sig,
                gpu_filt,
                mode,
                method,
            )

            key = self.cpu_version(cpu_sig, cpu_filt, mode, method)
            array_equal(output, key)

    @pytest.mark.benchmark(group="Convolve")
    @pytest.mark.parametrize("num_samps", [2**7, 2**10 + 1, 2**13])
    @pytest.mark.parametrize("num_taps", [125, 2**8, 2**13])
    @pytest.mark.parametrize("mode", ["full", "valid", "same"])
    @pytest.mark.parametrize("method", ["direct", "fft", "auto"])
    class TestConvolve:
        def cpu_version(self, sig, win, mode, method):
            return signal.convolve(sig, win, mode=mode, method=method)

        def gpu_version(self, sig, win, mode, method):
            with cp.cuda.Stream.null:
                out = cusignal.convolve(sig, win, mode=mode, method=method)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_convolve1d_cpu(
            self,
            rand_data_gen,
            benchmark,
            num_samps,
            num_taps,
            mode,
            method,
        ):
            cpu_sig, _ = rand_data_gen(num_samps, 1)
            cpu_win = signal.windows.hann(num_taps, 1)

            benchmark(self.cpu_version, cpu_sig, cpu_win, mode, method)

        def test_convolve1d_gpu(
            self,
            rand_data_gen,
            gpubenchmark,
            num_samps,
            num_taps,
            mode,
            method,
        ):

            cpu_sig, gpu_sig = rand_data_gen(num_samps, 1)
            gpu_win = cusignal.windows.hann(num_taps, 1)
            output = gpubenchmark(self.gpu_version, gpu_sig, gpu_win, mode, method)

            cpu_win = signal.windows.hann(num_taps, 1)
            key = self.cpu_version(cpu_sig, cpu_win, mode, method)
            array_equal(output, key)

    @pytest.mark.benchmark(group="FFTConvolve")
    @pytest.mark.parametrize("num_samps", [2**15])
    @pytest.mark.parametrize("mode", ["full", "valid", "same"])
    class TestFFTConvolve:
        def cpu_version(self, sig, mode):
            return signal.fftconvolve(sig, sig[::-1], mode=mode)

        def gpu_version(self, sig, mode):
            with cp.cuda.Stream.null:
                out = cusignal.fftconvolve(sig, sig[::-1], mode=mode)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_fftconvolve_cpu(self, rand_data_gen, benchmark, num_samps, mode):
            cpu_sig, _ = rand_data_gen(num_samps, 1)
            benchmark(self.cpu_version, cpu_sig, mode)

        def test_fftconvolve_gpu(self, rand_data_gen, gpubenchmark, num_samps, mode):

            cpu_sig, gpu_sig = rand_data_gen(num_samps, 1)
            output = gpubenchmark(self.gpu_version, gpu_sig, mode)

            key = self.cpu_version(cpu_sig, mode)
            array_equal(output, key)

    @pytest.mark.benchmark(group="Convolve2d")
    @pytest.mark.parametrize("num_samps", [2**8])
    @pytest.mark.parametrize("num_taps", [5, 100])
    @pytest.mark.parametrize("boundary", ["fill", "wrap", "symm"])
    @pytest.mark.parametrize("mode", ["full", "valid", "same"])
    class TestConvolve2d:
        def cpu_version(self, sig, filt, boundary, mode):
            return signal.convolve2d(sig, filt, boundary=boundary, mode=mode)

        def gpu_version(self, sig, filt, boundary, mode):
            with cp.cuda.Stream.null:
                out = cusignal.convolve2d(sig, filt, boundary=boundary, mode=mode)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_convolve2d_cpu(
            self,
            rand_data_gen,
            benchmark,
            num_samps,
            num_taps,
            boundary,
            mode,
        ):
            cpu_sig, _ = rand_data_gen(num_samps, 2)
            cpu_filt, _ = rand_data_gen(num_taps, 2)
            benchmark(self.cpu_version, cpu_sig, cpu_filt, boundary, mode)

        def test_convolve2d_gpu(
            self,
            rand_data_gen,
            gpubenchmark,
            num_samps,
            num_taps,
            boundary,
            mode,
        ):

            cpu_sig, gpu_sig = rand_data_gen(num_samps, 2)
            cpu_filt, gpu_filt = rand_data_gen(num_taps, 2)
            output = gpubenchmark(
                self.gpu_version,
                gpu_sig,
                gpu_filt,
                boundary,
                mode,
            )

            key = self.cpu_version(cpu_sig, cpu_filt, boundary, mode)
            array_equal(output, key)

    @pytest.mark.benchmark(group="Correlate2d")
    @pytest.mark.parametrize("num_samps", [2**8])
    @pytest.mark.parametrize("num_taps", [5, 100])
    @pytest.mark.parametrize("boundary", ["fill", "wrap", "symm"])
    @pytest.mark.parametrize("mode", ["full", "valid", "same"])
    class TestCorrelate2d:
        def cpu_version(self, sig, filt, boundary, mode):
            return signal.correlate2d(sig, filt, boundary=boundary, mode=mode)

        def gpu_version(self, sig, filt, boundary, mode):
            with cp.cuda.Stream.null:
                out = cusignal.correlate2d(sig, filt, boundary=boundary, mode=mode)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_correlate2d_cpu(
            self,
            rand_data_gen,
            benchmark,
            num_samps,
            num_taps,
            boundary,
            mode,
        ):
            cpu_sig, _ = rand_data_gen(num_samps, 2)
            cpu_filt, _ = rand_data_gen(num_taps, 2)
            benchmark(self.cpu_version, cpu_sig, cpu_filt, boundary, mode)

        def test_correlate2d_gpu(
            self,
            rand_data_gen,
            gpubenchmark,
            num_samps,
            num_taps,
            boundary,
            mode,
        ):

            cpu_sig, gpu_sig = rand_data_gen(num_samps, 2)
            cpu_filt, gpu_filt = rand_data_gen(num_taps, 2)
            output = gpubenchmark(
                self.gpu_version,
                gpu_sig,
                gpu_filt,
                boundary,
                mode,
            )

            key = self.cpu_version(cpu_sig, cpu_filt, boundary, mode)
            array_equal(output, key)
