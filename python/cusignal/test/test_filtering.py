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

from cusignal.test.utils import array_equal, _check_rapids_pytest_benchmark
from scipy import signal

gpubenchmark = _check_rapids_pytest_benchmark()


# Missing
# lfiltic
# detrend
# freq_shift


class TestFilter:
    @pytest.mark.benchmark(group="Wiener")
    @pytest.mark.parametrize("num_samps", [2 ** 15, 2 ** 24])
    class TestWiener:
        def cpu_version(self, cpu_sig):
            return signal.wiener(cpu_sig)

        @pytest.mark.cpu
        def test_wiener_cpu(self, rand_data_gen, benchmark, num_samps):
            cpu_sig, _ = rand_data_gen(num_samps)
            benchmark(self.cpu_version, cpu_sig)

        def test_wiener_gpu(self, rand_data_gen, gpubenchmark, num_samps):

            cpu_sig, gpu_sig = rand_data_gen(num_samps)
            output = gpubenchmark(cusignal.wiener, gpu_sig)

            key = self.cpu_version(cpu_sig)
            assert array_equal(cp.asnumpy(output), key)

    # @pytest.mark.benchmark(group="Lfiltic")
    # @pytest.mark.parametrize("num_a", [1.0, -1.0/3])
    # @pytest.mark.parametrize("num_b", [1.0/2, 1.0/4])
    # @pytest.mark.parametrize("num_y", [2.])

    # class TestLfiltic:
    #     #def cpu_version(self, a, b, y):
    #     def cpu_version(self, b, a, y):
    #         #return signal.lfiltic(a, b, y)
    #         return signal.lfiltic(b, a, y)

    #     @pytest.mark.cpu
    #     #def test_lfiltic_cpu(self, benchmark, num_a, num_b, num_y):
    #     def test_lfiltic_cpu(self, benchmark, num_b, num_a, num_y):

    #         #benchmark(self.cpu_version, num_a, num_b, num_y)
    #         benchmark(self.cpu_version, num_b, num_a, num_y)

    #     def test_lfiltic_gpu(self, gpubenchmark, num_b, num_a,num_y):
    #     #def test_lfiltic_gpu(self, gpubenchmark, num_a, num_b, num_y):
    #         d_num_a = cp.asarray(num_a)
    #         d_num_b = cp.asarray(num_b)
    #         d_num_y = cp.asarray(num_y)

    #         output = gpubenchmark(cusignal.lfiltic, d_num_b,d_num_a, d_num_y)
    #         #output = gpubenchmark(cusignal.lfiltic, d_num_a,num_b, num_y)

    #         key = self.cpu_version(num_b, num_a,  num_y)
    #         #key = self.cpu_version(num_a, num_b, num_y)
    #         assert array_equal(cp.asnumpy(output), key)

    @pytest.mark.benchmark(group="SOSFilt")
    @pytest.mark.parametrize("order", [32, 64])
    @pytest.mark.parametrize("num_samps", [2 ** 15, 2 ** 20])
    @pytest.mark.parametrize("num_signals", [1, 2, 10])
    @pytest.mark.parametrize("dtype", [np.float64])
    class TestSOSFilt:
        np.random.seed(1234)

        def cpu_version(self, sos, cpu_sig):
            return signal.sosfilt(sos, cpu_sig)

        @pytest.mark.cpu
        def test_sosfilt_cpu(
            self,
            rand_2d_data_gen,
            benchmark,
            num_signals,
            num_samps,
            order,
            dtype,
        ):
            cpu_sos = signal.ellip(order, 0.009, 80, 0.05, output="sos")
            cpu_sos = np.array(cpu_sos, dtype=dtype)
            cpu_sig = np.random.rand(num_signals, num_samps)
            cpu_sig = np.array(cpu_sig, dtype=dtype)
            benchmark(self.cpu_version, cpu_sos, cpu_sig)

        def test_sosfilt_gpu(
            self,
            rand_2d_data_gen,
            gpubenchmark,
            num_signals,
            num_samps,
            order,
            dtype,
        ):

            cpu_sos = signal.ellip(order, 0.009, 80, 0.05, output="sos")
            cpu_sos = np.array(cpu_sos, dtype=dtype)
            gpu_sos = cp.asarray(cpu_sos)
            cpu_sig = np.random.rand(num_signals, num_samps)
            cpu_sig = np.array(cpu_sig, dtype=dtype)
            gpu_sig = cp.asarray(cpu_sig)

            output = gpubenchmark(cusignal.sosfilt, gpu_sos, gpu_sig,)

            key = self.cpu_version(cpu_sos, cpu_sig)
            assert array_equal(cp.asnumpy(output), key)

    @pytest.mark.benchmark(group="Hilbert")
    @pytest.mark.parametrize("num_samps", [2 ** 15])
    class TestHilbert:
        def cpu_version(self, cpu_sig):
            return signal.hilbert(cpu_sig)

        @pytest.mark.cpu
        def test_hilbert_cpu(self, rand_data_gen, benchmark, num_samps):
            cpu_sig, _ = rand_data_gen(num_samps)
            benchmark(self.cpu_version, cpu_sig)

        def test_hilbert_gpu(self, rand_data_gen, gpubenchmark, num_samps):

            cpu_sig, gpu_sig = rand_data_gen(num_samps)
            output = gpubenchmark(cusignal.hilbert, gpu_sig)

            key = self.cpu_version(cpu_sig)
            assert array_equal(cp.asnumpy(output), key)

    @pytest.mark.benchmark(group="Hilbert2")
    @pytest.mark.parametrize("num_samps", [2 ** 8])
    class TestHilbert2:
        def cpu_version(self, cpu_sig):
            return signal.hilbert2(cpu_sig)

        @pytest.mark.cpu
        def test_hilbert2_cpu(self, rand_2d_data_gen, benchmark, num_samps):
            cpu_sig, _ = rand_2d_data_gen(num_samps)
            benchmark(self.cpu_version, cpu_sig)

        def test_hilbert2_gpu(self, rand_2d_data_gen, gpubenchmark, num_samps):

            cpu_sig, gpu_sig = rand_2d_data_gen(num_samps)
            output = gpubenchmark(cusignal.hilbert2, gpu_sig)

            key = self.cpu_version(cpu_sig)
            assert array_equal(cp.asnumpy(output), key)


    # @pytest.mark.benchmark(group="Detrend")
    # @pytest.mark.parametrize("randgen", np.random.RandomState(9))
    # @pytest.mark.parametrize("num_npoints", 1000)
    # @pytest.mark.parametrize("num_noise", randgen.randn(num_npoints))
    # @pytest.mark.parametrize("num_x", 3 + 2*np.linspace(0,1, num_npoints) + num_noise)

    # class TestDetrend:
    #     def cpu_version(self, x):
    #         return signal.detrend(x)

    #     @pytest.mark.cpu
    #     def test_detrend_cpu(self, benchmark, num_x):
    #         benchmark(self.cpu_version, num_x)

    #     def test_detrend_gpu(self, gpubenchmark, num_x):

    #         output = gpubenchmark(cusignal.detrend, num_x)

    #         key = self.cpu_version(num_x)
    #         assert array_equal(cp.asnumpy(output), key)

    # @pytest.mark.benchmark(group="FreqShift")
    # @pytest.mark.parametrize("freq", np.fft.fftfreq(10, 0.1))
    # class TestFreqShift:
    #     def cpu_version(self, cpu_sig, freq):
    #         return signal.freq_shift(freq)

    #     @pytest.mark.cpu
    #     def test_freq_shift_cpu(self, benchmark, freq):
    #         benchmark(self.cpu_version, freq)

    #     def test_freq_shift_gpu(self, gpubenchmark):

    #         output = gpubenchmark(cusignal.detrend, freq)

    #         key = self.cpu_version(freq)
    #         assert array_equal(cp.asnumpy(output), key)

    @pytest.mark.benchmark(group="Decimate")
    @pytest.mark.parametrize("num_samps", [2 ** 14, 2 ** 18])
    @pytest.mark.parametrize("downsample_factor", [2, 3, 4, 8, 64])
    @pytest.mark.parametrize("zero_phase", [True, False])
    class TestDecimate:
        def cpu_version(self, cpu_sig, downsample_factor, zero_phase):
            return signal.decimate(
                cpu_sig, downsample_factor, ftype="fir", zero_phase=zero_phase
            )

        @pytest.mark.cpu
        def test_decimate_cpu(
            self,
            benchmark,
            linspace_data_gen,
            num_samps,
            downsample_factor,
            zero_phase,
        ):
            cpu_sig, _ = linspace_data_gen(0, 10, num_samps, endpoint=False)
            benchmark(self.cpu_version, cpu_sig, downsample_factor, zero_phase)

        def test_decimate_gpu(
            self,
            gpubenchmark,
            linspace_data_gen,
            num_samps,
            downsample_factor,
            zero_phase,
        ):
            cpu_sig, gpu_sig = linspace_data_gen(
                0, 10, num_samps, endpoint=False
            )
            output = gpubenchmark(
                cusignal.decimate,
                gpu_sig,
                downsample_factor,
                zero_phase=zero_phase,
            )

            key = self.cpu_version(cpu_sig, downsample_factor, zero_phase)
            assert array_equal(cp.asnumpy(output), key)

    @pytest.mark.benchmark(group="Resample")
    @pytest.mark.parametrize("num_samps", [2 ** 14])
    @pytest.mark.parametrize("resample_num_samps", [2 ** 12, 2 ** 16])
    @pytest.mark.parametrize("window", [("kaiser", 0.5)])
    class TestResample:
        def cpu_version(self, cpu_sig, resample_num_samps, window):
            return signal.resample(cpu_sig, resample_num_samps, window=window)

        @pytest.mark.cpu
        def test_resample_cpu(
            self,
            linspace_data_gen,
            benchmark,
            num_samps,
            resample_num_samps,
            window,
        ):
            cpu_sig, _ = linspace_data_gen(0, 10, num_samps, endpoint=False)
            benchmark(
                self.cpu_version, cpu_sig, resample_num_samps, window,
            )

        def test_resample_gpu(
            self,
            linspace_data_gen,
            gpubenchmark,
            num_samps,
            resample_num_samps,
            window,
        ):

            cpu_sig, gpu_sig = linspace_data_gen(
                0, 10, num_samps, endpoint=False
            )
            output = gpubenchmark(
                cusignal.resample, gpu_sig, resample_num_samps, window=window
            )

            key = self.cpu_version(cpu_sig, resample_num_samps, window)
            assert array_equal(cp.asnumpy(output), key)

    @pytest.mark.benchmark(group="ResamplePoly")
    @pytest.mark.parametrize("num_samps", [2 ** 14])
    @pytest.mark.parametrize("up", [2, 3, 7])
    @pytest.mark.parametrize("down", [1, 2, 9])
    @pytest.mark.parametrize("window", [("kaiser", 0.5)])
    class TestResamplePoly:
        def cpu_version(self, cpu_sig, up, down, window):
            return signal.resample_poly(cpu_sig, up, down, window=window)

        @pytest.mark.cpu
        def test_resample_poly_cpu(
            self, linspace_data_gen, benchmark, num_samps, up, down, window
        ):
            cpu_sig, _ = linspace_data_gen(0, 10, num_samps, endpoint=False)
            benchmark(
                self.cpu_version, cpu_sig, up, down, window,
            )

        def test_resample_poly_gpu(
            self, linspace_data_gen, gpubenchmark, num_samps, up, down, window,
        ):

            cpu_sig, gpu_sig = linspace_data_gen(
                0, 10, num_samps, endpoint=False
            )
            output = gpubenchmark(
                cusignal.resample_poly, gpu_sig, up, down, window=window,
            )

            key = self.cpu_version(cpu_sig, up, down, window)
            assert array_equal(cp.asnumpy(output), key)

    @pytest.mark.benchmark(group="UpFirDn")
    @pytest.mark.parametrize("num_samps", [2 ** 14])
    @pytest.mark.parametrize("up", [2, 3, 7])
    @pytest.mark.parametrize("down", [1, 2, 9])
    @pytest.mark.parametrize("axis", [-1, 0])
    class TestUpFirDn:
        def cpu_version(self, cpu_sig, up, down, axis):
            return signal.upfirdn([1, 1, 1], cpu_sig, up, down, axis)

        @pytest.mark.cpu
        def test_upfirdn_cpu(
            self, rand_data_gen, benchmark, num_samps, up, down, axis
        ):
            cpu_sig, _ = rand_data_gen(num_samps)
            benchmark(
                self.cpu_version, cpu_sig, up, down, axis,
            )

        def test_upfirdn_gpu(
            self, rand_data_gen, gpubenchmark, num_samps, up, down, axis,
        ):

            cpu_sig, gpu_sig = rand_data_gen(num_samps)
            output = gpubenchmark(
                cusignal.upfirdn, [1, 1, 1], gpu_sig, up, down, axis,
            )

            key = self.cpu_version(cpu_sig, up, down, axis)
            assert array_equal(cp.asnumpy(output), key)

    @pytest.mark.benchmark(group="UpFirDn2d")
    @pytest.mark.parametrize("num_samps", [2 ** 8])
    @pytest.mark.parametrize("up", [2, 3, 7])
    @pytest.mark.parametrize("down", [1, 2, 9])
    @pytest.mark.parametrize("axis", [-1, 0])
    class TestUpFirDn2d:
        def cpu_version(self, cpu_sig, up, down, axis):
            return signal.upfirdn([1, 1, 1], cpu_sig, up, down, axis)

        @pytest.mark.cpu
        def test_upfirdn2d_cpu(
            self, rand_2d_data_gen, benchmark, num_samps, up, down, axis
        ):
            cpu_sig, _ = rand_2d_data_gen(num_samps)
            benchmark(
                self.cpu_version, cpu_sig, up, down, axis,
            )

        def test_upfirdn2d_gpu(
            self, rand_2d_data_gen, gpubenchmark, num_samps, up, down, axis,
        ):

            cpu_sig, gpu_sig = rand_2d_data_gen(num_samps)
            output = gpubenchmark(
                cusignal.upfirdn, [1, 1, 1], gpu_sig, up, down, axis,
            )

            key = self.cpu_version(cpu_sig, up, down, axis)
            assert array_equal(cp.asnumpy(output), key)

    @pytest.mark.benchmark(group="Firfilter")
    @pytest.mark.parametrize("num_samps", [2 ** 14, 2 ** 18])
    @pytest.mark.parametrize("filter_len", [8, 32, 128])
    class TestFirfilter:
        def cpu_version(self, cpu_sig, cpu_filter):
            return signal.lfilter(
                cpu_filter, 1, cpu_sig
            )

        @pytest.mark.cpu
        def test_firfilter_cpu(
            self,
            benchmark,
            linspace_data_gen,
            num_samps,
            filter_len,
        ):
            cpu_sig, _ = linspace_data_gen(0, 10, num_samps, endpoint=False)
            cpu_filter, _ = signal.butter(filter_len, 0.5)
            benchmark(self.cpu_version, cpu_sig, cpu_filter)

        def test_firfilter_gpu(
            self,
            gpubenchmark,
            linspace_data_gen,
            num_samps,
            filter_len,
        ):
            cpu_sig, gpu_sig = linspace_data_gen(
                0, 10, num_samps, endpoint=False
            )
            cpu_filter, _ = signal.butter(filter_len, 0.5)
            gpu_filter = cp.asarray(cpu_filter)
            output = gpubenchmark(
                cusignal.firfilter,
                gpu_filter,
                gpu_sig,
            )

            key = self.cpu_version(cpu_sig, cpu_filter)
            assert array_equal(cp.asnumpy(output), key)
