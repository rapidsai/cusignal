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


class TestWindows:
    @pytest.mark.benchmark(group="GeneralCosine")
    @pytest.mark.parametrize("num_samps", [2**15])
    class TestGeneralCosine:
        def cpu_version(self, num_samps, arr):
            return signal.windows.general_cosine(num_samps, arr, sym=False)

        def gpu_version(self, num_samps, arr):
            with cp.cuda.Stream.null:
                out = cusignal.windows.general_cosine(num_samps, arr, sym=False)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_general_cosine_cpu(self, benchmark, num_samps):
            HFT90D = [1, 1.942604, 1.340318, 0.440811, 0.043097]

            benchmark(self.cpu_version, num_samps, HFT90D)

        def test_general_cosine_gpu(self, gpubenchmark, num_samps):
            HFT90D = [1, 1.942604, 1.340318, 0.440811, 0.043097]

            output = gpubenchmark(self.cpu_version, num_samps, HFT90D)

            key = self.cpu_version(num_samps, HFT90D)
            array_equal(output, key)

    @pytest.mark.benchmark(group="Boxcar")
    @pytest.mark.parametrize("num_samps", [2**15])
    class TestBoxcar:
        def cpu_version(self, num_samps):
            return signal.windows.boxcar(num_samps)

        def gpu_version(self, num_samps):
            with cp.cuda.Stream.null:
                out = cusignal.windows.boxcar(num_samps)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_boxcar_cpu(self, benchmark, num_samps):
            benchmark(self.cpu_version, num_samps)

        def test_boxcar_gpu(self, gpubenchmark, num_samps):
            output = gpubenchmark(self.gpu_version, num_samps)

            key = self.cpu_version(num_samps)
            array_equal(output, key)

    @pytest.mark.benchmark(group="Triang")
    @pytest.mark.parametrize("num_samps", [2**15, 2**15 - 1])
    class TestTriang:
        def cpu_version(self, num_samps):
            return signal.windows.triang(num_samps)

        def gpu_version(self, num_samps):
            with cp.cuda.Stream.null:
                out = cusignal.windows.triang(num_samps)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_triang_cpu(self, benchmark, num_samps):
            benchmark(self.cpu_version, num_samps)

        def test_triang_gpu(self, gpubenchmark, num_samps):
            output = gpubenchmark(self.gpu_version, num_samps)

            key = self.cpu_version(num_samps)
            array_equal(output, key)

    @pytest.mark.benchmark(group="Parzen")
    @pytest.mark.parametrize("num_samps", [2**15, 2**15 - 1])
    class TestParzen:
        def cpu_version(self, num_samps):
            return signal.windows.parzen(num_samps)

        def gpu_version(self, num_samps):
            with cp.cuda.Stream.null:
                out = cusignal.windows.parzen(num_samps)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_parzen_cpu(self, benchmark, num_samps):
            benchmark(self.cpu_version, num_samps)

        def test_parzen_gpu(self, gpubenchmark, num_samps):
            output = gpubenchmark(self.gpu_version, num_samps)

            key = self.cpu_version(num_samps)
            array_equal(output, key)

    @pytest.mark.benchmark(group="Bohman")
    @pytest.mark.parametrize("num_samps", [2**15])
    class TestBohman:
        def cpu_version(self, num_samps):
            return signal.windows.bohman(num_samps)

        def gpu_version(self, num_samps):
            with cp.cuda.Stream.null:
                out = cusignal.windows.bohman(num_samps)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_bohman_cpu(self, benchmark, num_samps):
            benchmark(self.cpu_version, num_samps)

        def test_bohman_gpu(self, gpubenchmark, num_samps):
            output = gpubenchmark(self.gpu_version, num_samps)

            key = self.cpu_version(num_samps)
            array_equal(output, key)

    @pytest.mark.benchmark(group="Blackman")
    @pytest.mark.parametrize("num_samps", [2**15])
    class TestBlackman:
        def cpu_version(self, num_samps):
            return signal.windows.blackman(num_samps)

        def gpu_version(self, num_samps):
            with cp.cuda.Stream.null:
                out = cusignal.windows.blackman(num_samps)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_blackman_cpu(self, benchmark, num_samps):
            benchmark(self.cpu_version, num_samps)

        def test_blackman_gpu(self, gpubenchmark, num_samps):
            output = gpubenchmark(self.gpu_version, num_samps)

            key = self.cpu_version(num_samps)
            array_equal(output, key)

    @pytest.mark.benchmark(group="Nuttall")
    @pytest.mark.parametrize("num_samps", [2**15])
    class TestNuttall:
        def cpu_version(self, num_samps):
            return signal.windows.nuttall(num_samps)

        def gpu_version(self, num_samps):
            with cp.cuda.Stream.null:
                out = cusignal.windows.nuttall(num_samps)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_nuttall_cpu(self, benchmark, num_samps):
            benchmark(self.cpu_version, num_samps)

        def test_nuttall_gpu(self, gpubenchmark, num_samps):
            output = gpubenchmark(self.gpu_version, num_samps)

            key = self.cpu_version(num_samps)
            array_equal(output, key)

    @pytest.mark.benchmark(group="BlackmanHarris")
    @pytest.mark.parametrize("num_samps", [2**15])
    class TestBlackmanHarris:
        def cpu_version(self, num_samps):
            return signal.windows.blackmanharris(num_samps)

        def gpu_version(self, num_samps):
            with cp.cuda.Stream.null:
                out = cusignal.windows.blackmanharris(num_samps)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_blackmanharris_cpu(self, benchmark, num_samps):
            benchmark(self.cpu_version, num_samps)

        def test_blackmanharris_gpu(self, gpubenchmark, num_samps):
            output = gpubenchmark(self.gpu_version, num_samps)

            key = self.cpu_version(num_samps)
            array_equal(output, key)

    @pytest.mark.benchmark(group="FlatTop")
    @pytest.mark.parametrize("num_samps", [2**15])
    class TestFlatTop:
        def cpu_version(self, num_samps):
            return signal.windows.flattop(num_samps)

        def gpu_version(self, num_samps):
            with cp.cuda.Stream.null:
                out = cusignal.windows.flattop(num_samps)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_flattop_cpu(self, benchmark, num_samps):
            benchmark(self.cpu_version, num_samps)

        def test_flattop_gpu(self, gpubenchmark, num_samps):
            output = gpubenchmark(self.gpu_version, num_samps)

            key = self.cpu_version(num_samps)
            array_equal(output, key)

    @pytest.mark.benchmark(group="Barlett")
    @pytest.mark.parametrize("num_samps", [2**15])
    class TestBarlett:
        def cpu_version(self, num_samps):
            return signal.windows.bartlett(num_samps)

        def gpu_version(self, num_samps):
            with cp.cuda.Stream.null:
                out = cusignal.windows.bartlett(num_samps)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_bartlett_cpu(self, benchmark, num_samps):
            benchmark(self.cpu_version, num_samps)

        def test_bartlett_gpu(self, gpubenchmark, num_samps):
            output = gpubenchmark(self.gpu_version, num_samps)

            key = self.cpu_version(num_samps)
            array_equal(output, key)

    @pytest.mark.benchmark(group="Tukey")
    @pytest.mark.parametrize("num_samps", [2**15])
    @pytest.mark.parametrize("alpha", [0.25, 0.5])
    class TestTukey:
        def cpu_version(self, num_samps, alpha):
            return signal.windows.tukey(num_samps, alpha, sym=True)

        def gpu_version(self, num_samps, alpha):
            with cp.cuda.Stream.null:
                out = cusignal.windows.tukey(num_samps, alpha, sym=True)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_tukey_cpu(self, benchmark, num_samps, alpha):
            benchmark(self.cpu_version, num_samps, alpha)

        def test_tukey_gpu(self, gpubenchmark, num_samps, alpha):
            output = gpubenchmark(self.gpu_version, num_samps, alpha)

            key = self.cpu_version(num_samps, alpha)
            array_equal(output, key)

    @pytest.mark.benchmark(group="BartHann")
    @pytest.mark.parametrize("num_samps", [2**15])
    class TestBartHann:
        def cpu_version(self, num_samps):
            return signal.windows.barthann(num_samps)

        def gpu_version(self, num_samps):
            with cp.cuda.Stream.null:
                out = cusignal.windows.barthann(num_samps)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_barthann_cpu(self, benchmark, num_samps):
            benchmark(self.cpu_version, num_samps)

        def test_barthann_gpu(self, gpubenchmark, num_samps):
            output = gpubenchmark(self.gpu_version, num_samps)

            key = self.cpu_version(num_samps)
            array_equal(output, key)

    @pytest.mark.benchmark(group="GeneralHamming")
    @pytest.mark.parametrize("num_samps", [2**15])
    @pytest.mark.parametrize("alpha", [0.25, 0.5])
    class TestGeneralHamming:
        def cpu_version(self, num_samps, alpha):
            return signal.windows.general_hamming(num_samps, alpha, sym=True)

        def gpu_version(self, num_samps, alpha):
            with cp.cuda.Stream.null:
                out = cusignal.windows.general_hamming(num_samps, alpha, sym=True)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_general_hamming_cpu(self, benchmark, num_samps, alpha):
            benchmark(self.cpu_version, num_samps, alpha)

        def test_general_hamming_gpu(self, gpubenchmark, num_samps, alpha):
            output = gpubenchmark(self.gpu_version, num_samps, alpha)

            key = self.cpu_version(num_samps, alpha)
            array_equal(output, key)

    @pytest.mark.benchmark(group="Hamming")
    @pytest.mark.parametrize("num_samps", [2**15])
    class TestHamming:
        def cpu_version(self, num_samps):
            return signal.windows.hamming(num_samps)

        def gpu_version(self, num_samps):
            with cp.cuda.Stream.null:
                out = cusignal.windows.hamming(num_samps)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_hamming_cpu(self, benchmark, num_samps):
            benchmark(self.cpu_version, num_samps)

        def test_hamming_gpu(self, gpubenchmark, num_samps):
            output = gpubenchmark(self.gpu_version, num_samps)

            key = self.cpu_version(num_samps)
            array_equal(output, key)

    @pytest.mark.benchmark(group="Kaiser")
    @pytest.mark.parametrize("num_samps", [2**15])
    @pytest.mark.parametrize("beta", [0.25, 0.5])
    class TestKaiser:
        def cpu_version(self, num_samps, beta):
            return signal.windows.kaiser(num_samps, beta, sym=True)

        def gpu_version(self, num_samps, beta):
            with cp.cuda.Stream.null:
                out = cusignal.windows.kaiser(num_samps, beta, sym=True)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_kaiser_cpu(self, benchmark, num_samps, beta):
            benchmark(self.cpu_version, num_samps, beta)

        def test_kaiser_gpu(self, gpubenchmark, num_samps, beta):
            output = gpubenchmark(self.gpu_version, num_samps, beta)

            key = self.cpu_version(num_samps, beta)
            array_equal(output, key)

    @pytest.mark.benchmark(group="Gaussian")
    @pytest.mark.parametrize("num_samps", [2**15])
    @pytest.mark.parametrize("std", [3, 7])
    class TestGaussian:
        def cpu_version(self, num_samps, std):
            return signal.windows.gaussian(num_samps, std)

        def gpu_version(self, num_samps, std):
            with cp.cuda.Stream.null:
                out = cusignal.windows.gaussian(num_samps, std)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_gaussian_cpu(self, benchmark, num_samps, std):
            benchmark(self.cpu_version, num_samps, std)

        def test_gaussian_gpu(self, gpubenchmark, num_samps, std):
            output = gpubenchmark(self.gpu_version, num_samps, std)

            key = self.cpu_version(num_samps, std)
            array_equal(output, key)

    @pytest.mark.benchmark(group="GeneralGaussian")
    @pytest.mark.parametrize("num_samps", [2**15])
    @pytest.mark.parametrize("p", [0.75, 1.5])
    @pytest.mark.parametrize("std", [3, 7])
    class TestGeneralGaussian:
        def cpu_version(self, num_samps, p, std):
            return signal.windows.general_gaussian(num_samps, p, std)

        def gpu_version(self, num_samps, p, std):
            with cp.cuda.Stream.null:
                out = cusignal.windows.general_gaussian(num_samps, p, std)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_general_gaussian_cpu(self, benchmark, num_samps, p, std):
            benchmark(self.cpu_version, num_samps, p, std)

        def test_general_gaussian_gpu(self, gpubenchmark, num_samps, p, std):
            output = gpubenchmark(self.gpu_version, num_samps, p, std)

            key = self.cpu_version(num_samps, p, std)
            array_equal(output, key)

    @pytest.mark.benchmark(group="Chebwin")
    @pytest.mark.parametrize("num_samps", [2**15, 2**15 - 1])
    @pytest.mark.parametrize("at", [50, 100])
    class TestChebwin:
        def cpu_version(self, num_samps, at):
            return signal.windows.chebwin(num_samps, at)

        def gpu_version(self, num_samps, at):
            with cp.cuda.Stream.null:
                out = cusignal.windows.chebwin(num_samps, at)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_chebwin_cpu(self, benchmark, num_samps, at):
            benchmark(self.cpu_version, num_samps, at)

        def test_chebwin_gpu(self, gpubenchmark, num_samps, at):
            output = gpubenchmark(self.gpu_version, num_samps, at)

            key = self.cpu_version(num_samps, at)
            array_equal(output, key)

    @pytest.mark.benchmark(group="Cosine")
    @pytest.mark.parametrize("num_samps", [2**15])
    class TestCosine:
        def cpu_version(self, num_samps):
            return signal.windows.cosine(num_samps)

        def gpu_version(self, num_samps):
            with cp.cuda.Stream.null:
                out = cusignal.windows.cosine(num_samps)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_cosine_cpu(self, benchmark, num_samps):
            benchmark(self.cpu_version, num_samps)

        def test_cosine_gpu(self, gpubenchmark, num_samps):
            output = gpubenchmark(self.gpu_version, num_samps)

            key = self.cpu_version(num_samps)
            array_equal(output, key)

    @pytest.mark.benchmark(group="Exponential")
    @pytest.mark.parametrize("num_samps", [2**15])
    @pytest.mark.parametrize("tau", [1.5, 3.0])
    class TestExponential:
        def cpu_version(self, num_samps, tau):
            return signal.windows.exponential(num_samps, tau=tau)

        def gpu_version(self, num_samps, tau):
            with cp.cuda.Stream.null:
                out = cusignal.windows.exponential(num_samps, tau=tau)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_exponential_cpu(self, benchmark, num_samps, tau):
            benchmark(self.cpu_version, num_samps, tau)

        def test_exponential_gpu(self, gpubenchmark, num_samps, tau):
            output = gpubenchmark(self.gpu_version, num_samps, tau)

            key = self.cpu_version(num_samps, tau)
            array_equal(output, key)

    @pytest.mark.benchmark(group="Taylor")
    @pytest.mark.parametrize("num_samps", [2**15])
    @pytest.mark.parametrize("nbar", [20, 100])
    @pytest.mark.parametrize("norm", [True, False])
    class TestTaylor:
        def cpu_version(self, num_samps, nbar, norm):
            return signal.windows.taylor(num_samps, nbar, norm=norm)

        def gpu_version(self, num_samps, nbar, norm):
            with cp.cuda.Stream.null:
                out = cusignal.windows.taylor(num_samps, nbar, norm=norm)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_taylor_cpu(self, benchmark, num_samps, nbar, norm):
            benchmark(self.cpu_version, num_samps, nbar, norm)

        def test_taylor_gpu(self, gpubenchmark, num_samps, nbar, norm):
            output = gpubenchmark(self.gpu_version, num_samps, nbar, norm)

            key = self.cpu_version(num_samps, nbar, norm)
            array_equal(output, key)

    @pytest.mark.benchmark(group="GetWindow")
    @pytest.mark.parametrize("window", ["triang", "boxcar", "nuttall"])
    @pytest.mark.parametrize("num_samps", [2**15])
    class TestGetWindow:
        def cpu_version(self, window, num_samps):
            return signal.windows.get_window(window, num_samps)

        def gpu_version(self, window, num_samps):
            with cp.cuda.Stream.null:
                out = cusignal.windows.get_window(window, num_samps)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_get_window_cpu(self, benchmark, window, num_samps):
            benchmark(self.cpu_version, window, num_samps)

        def test_get_window_gpu(self, gpubenchmark, window, num_samps):
            output = gpubenchmark(self.gpu_version, window, num_samps)

            key = self.cpu_version(window, num_samps)
            array_equal(output, key)
