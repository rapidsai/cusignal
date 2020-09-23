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
# freq_shift


class TestFilter:
    @pytest.mark.benchmark(group="Wiener")
    @pytest.mark.parametrize("num_samps", [2 ** 15, 2 ** 24])
    class TestWiener:
        def cpu_version(self, sig):
            return signal.wiener(sig)

        def gpu_version(self, sig):
            with cp.cuda.Stream.null:
                out = cusignal.wiener(sig)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_wiener_cpu(self, rand_data_gen, benchmark, num_samps):
            cpu_sig, _ = rand_data_gen(num_samps)
            benchmark(self.cpu_version, cpu_sig)

        def test_wiener_gpu(self, rand_data_gen, gpubenchmark, num_samps):

            cpu_sig, gpu_sig = rand_data_gen(num_samps)
            output = gpubenchmark(self.gpu_version, gpu_sig)

            key = self.cpu_version(cpu_sig)
            assert array_equal(cp.asnumpy(output), key)

    # @pytest.mark.benchmark(group="Lfiltic")
    # @pytest.mark.parametrize("num_a", [1.0, 1.0/3])
    # @pytest.mark.parametrize("num_b", [1.0/2, 1.0/4])
    # @pytest.mark.parametrize("num_y", [-1.])

    # class TestLfiltic:
    #     def cpu_version(self, b, a, y):
    #         return signal.lfiltic(b, a, y)

    #     def gpu_version(self, b, a, y):
    #         with cp.cuda.Stream.null:
    #             out = cusignal.lfiltic(b, a, y)
    #         cp.cuda.Stream.null.synchronize()
    #         return out

    #     @pytest.mark.cpu
    #     def test_lfiltic_cpu(self, benchmark, num_b, num_a, num_y):

    #         benchmark(self.cpu_version, num_b, num_a, num_y)

    #     def test_lfiltic_gpu(self, gpubenchmark, num_b, num_a,num_y):

    #         d_num_a = cp.asarray(num_a)
    #         d_num_b = cp.asarray(num_b)
    #         output = gpubenchmark(self.gpu_version, d_num_b, d_num_a, num_y)

    #         key = self.cpu_version(num_b, num_a,  num_y)
    #         assert array_equal(cp.asnumpy(output), key)

    @pytest.mark.benchmark(group="SOSFilt")
    @pytest.mark.parametrize("order", [32, 64])
    @pytest.mark.parametrize("num_samps", [2 ** 15, 2 ** 20])
    @pytest.mark.parametrize("num_signals", [1, 2, 10])
    @pytest.mark.parametrize("dtype", [np.float64])
    class TestSOSFilt:
        np.random.seed(1234)

        def cpu_version(self, sos, sig):
            return signal.sosfilt(sos, sig)

        def gpu_version(self, sos, sig):
            with cp.cuda.Stream.null:
                out = cusignal.sosfilt(sos, sig)
            cp.cuda.Stream.null.synchronize()
            return out

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

            output = gpubenchmark(
                self.gpu_version,
                gpu_sos,
                gpu_sig,
            )

            key = self.cpu_version(cpu_sos, cpu_sig)
            assert array_equal(cp.asnumpy(output), key)

    @pytest.mark.benchmark(group="Hilbert")
    @pytest.mark.parametrize("num_samps", [2 ** 15])
    class TestHilbert:
        def cpu_version(self, sig):
            return signal.hilbert(sig)

        def gpu_version(self, sig):
            with cp.cuda.Stream.null:
                out = cusignal.hilbert(sig)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_hilbert_cpu(self, rand_data_gen, benchmark, num_samps):
            cpu_sig, _ = rand_data_gen(num_samps)
            benchmark(self.cpu_version, cpu_sig)

        def test_hilbert_gpu(self, rand_data_gen, gpubenchmark, num_samps):

            cpu_sig, gpu_sig = rand_data_gen(num_samps)
            output = gpubenchmark(self.gpu_version, gpu_sig)

            key = self.cpu_version(cpu_sig)
            assert array_equal(cp.asnumpy(output), key)

    @pytest.mark.benchmark(group="Hilbert2")
    @pytest.mark.parametrize("num_samps", [2 ** 8])
    class TestHilbert2:
        def cpu_version(self, sig):
            return signal.hilbert2(sig)

        def gpu_version(self, sig):
            with cp.cuda.Stream.null:
                out = cusignal.hilbert2(sig)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_hilbert2_cpu(self, rand_2d_data_gen, benchmark, num_samps):
            cpu_sig, _ = rand_2d_data_gen(num_samps)
            benchmark(self.cpu_version, cpu_sig)

        def test_hilbert2_gpu(self, rand_2d_data_gen, gpubenchmark, num_samps):

            cpu_sig, gpu_sig = rand_2d_data_gen(num_samps)
            output = gpubenchmark(self.gpu_version, gpu_sig)

            key = self.cpu_version(cpu_sig)
            assert array_equal(cp.asnumpy(output), key)

    @pytest.mark.benchmark(group="Detrend")
    @pytest.mark.parametrize("num_samps", [2 ** 8])
    class TestDetrend:
        def cpu_version(self, sig):
            return signal.detrend(sig)

        def gpu_version(self, sig):
            with cp.cuda.Stream.null:
                out = cusignal.detrend(sig)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_detrend_cpu(self, linspace_data_gen, benchmark, num_samps):
            cpu_sig, _ = linspace_data_gen(0, 10, num_samps)
            benchmark(self.cpu_version, cpu_sig)

        def test_detrend_gpu(self, linspace_data_gen, gpubenchmark, num_samps):

            cpu_sig, gpu_sig = linspace_data_gen(0, 10, num_samps)
            output = gpubenchmark(cusignal.detrend, gpu_sig)

            key = self.cpu_version(cpu_sig)
            assert array_equal(cp.asnumpy(output), key)

    # @pytest.mark.benchmark(group="FreqShift")
    # @pytest.mark.parametrize("num_samps", [2 ** 8])
    # @pytest.mark.parametrize("freq", np.fft.fftfreq(10, 0.1))
    # @pytest.mark.parametrize("fs", [0.1])
    # class TestFreqShift:
    #     def cpu_version(self, freq, fs, num_samps):
    #         return np.fftshift(freq, fs, num_samps)

    #     def gpu_version(self, freq, fs, num_samps):
    #         with cp.cuda.Stream.null:
    #             out = cusignal.freq_shift(freq, fs, num_samps)
    #         cp.cuda.Stream.null.synchronize()
    #         return out

    #     @pytest.mark.cpu
    #     def test_freq_shift_cpu(
    #         self, rand_complex_data_gen, benchmark, freq, fs, num_samps
    #     ):
    #         cpu_sig, _ = rand_complex_data_gen(num_samps)
    #         benchmark(self.cpu_version, cpu_sig, freq, fs)

    #     def test_freq_shift_gpu(
    #         self, rand_complex_data_gen, gpubenchmark, freq, fs, num_samps
    #     ):
    #         cpu_sig, gpu_sig = rand_complex_data_gen(num_samps)
    #         output = gpubenchmark(self.gpu_version, gpu_sig, freq, fs)

    #         key = self.cpu_version(cpu_sig, freq, fs)
    #         assert array_equal(cp.asnumpy(output), key)

    @pytest.mark.benchmark(group="Decimate")
    @pytest.mark.parametrize("num_samps", [2 ** 14, 2 ** 18])
    @pytest.mark.parametrize("downsample_factor", [2, 3, 4, 8, 64])
    @pytest.mark.parametrize("zero_phase", [True, False])
    class TestDecimate:
        def cpu_version(self, sig, downsample_factor, zero_phase):
            return signal.decimate(
                sig, downsample_factor, ftype="fir", zero_phase=zero_phase
            )

        def gpu_version(self, sig, downsample_factor, zero_phase):
            with cp.cuda.Stream.null:
                out = cusignal.decimate(
                    sig, downsample_factor, zero_phase=zero_phase
                )
            cp.cuda.Stream.null.synchronize()
            return out

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
                self.gpu_version,
                gpu_sig,
                downsample_factor,
                zero_phase,
            )

            key = self.cpu_version(cpu_sig, downsample_factor, zero_phase)
            assert array_equal(cp.asnumpy(output), key)

    @pytest.mark.benchmark(group="Resample")
    @pytest.mark.parametrize("num_samps", [2 ** 14])
    @pytest.mark.parametrize("resample_num_samps", [2 ** 12, 2 ** 16])
    @pytest.mark.parametrize("window", [("kaiser", 0.5)])
    class TestResample:
        def cpu_version(self, sig, resample_num_samps, window):
            return signal.resample(sig, resample_num_samps, window=window)

        def gpu_version(self, sig, resample_num_samps, window):
            with cp.cuda.Stream.null:
                out = cusignal.resample(sig, resample_num_samps, window=window)
            cp.cuda.Stream.null.synchronize()
            return out

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
                self.cpu_version,
                cpu_sig,
                resample_num_samps,
                window,
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
                self.gpu_version, gpu_sig, resample_num_samps, window
            )

            key = self.cpu_version(cpu_sig, resample_num_samps, window)
            assert array_equal(cp.asnumpy(output), key)

    @pytest.mark.benchmark(group="ResamplePoly")
    @pytest.mark.parametrize("num_samps", [2 ** 14])
    @pytest.mark.parametrize("up", [2, 3, 7])
    @pytest.mark.parametrize("down", [1, 2, 9])
    @pytest.mark.parametrize("window", [("kaiser", 0.5)])
    class TestResamplePoly:
        def cpu_version(self, sig, up, down, window):
            return signal.resample_poly(sig, up, down, window=window)

        def gpu_version(self, sig, up, down, window):
            with cp.cuda.Stream.null:
                out = cusignal.resample_poly(sig, up, down, window=window)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_resample_poly_cpu(
            self, linspace_data_gen, benchmark, num_samps, up, down, window
        ):
            cpu_sig, _ = linspace_data_gen(0, 10, num_samps, endpoint=False)
            benchmark(
                self.cpu_version,
                cpu_sig,
                up,
                down,
                window,
            )

        def test_resample_poly_gpu(
            self,
            linspace_data_gen,
            gpubenchmark,
            num_samps,
            up,
            down,
            window,
        ):

            cpu_sig, gpu_sig = linspace_data_gen(
                0, 10, num_samps, endpoint=False
            )
            output = gpubenchmark(
                self.gpu_version,
                gpu_sig,
                up,
                down,
                window,
            )

            key = self.cpu_version(cpu_sig, up, down, window)
            assert array_equal(cp.asnumpy(output), key)

    @pytest.mark.benchmark(group="UpFirDn")
    @pytest.mark.parametrize("num_samps", [2 ** 14])
    @pytest.mark.parametrize("up", [2, 3, 7])
    @pytest.mark.parametrize("down", [1, 2, 9])
    @pytest.mark.parametrize("axis", [-1, 0])
    class TestUpFirDn:
        def cpu_version(self, sig, up, down, axis):
            return signal.upfirdn([1, 1, 1], sig, up, down, axis)

        def gpu_version(self, sig, up, down, axis):
            with cp.cuda.Stream.null:
                out = cusignal.upfirdn([1, 1, 1], sig, up, down, axis)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_upfirdn_cpu(
            self, rand_data_gen, benchmark, num_samps, up, down, axis
        ):
            cpu_sig, _ = rand_data_gen(num_samps)
            benchmark(
                self.cpu_version,
                cpu_sig,
                up,
                down,
                axis,
            )

        def test_upfirdn_gpu(
            self,
            rand_data_gen,
            gpubenchmark,
            num_samps,
            up,
            down,
            axis,
        ):

            cpu_sig, gpu_sig = rand_data_gen(num_samps)
            output = gpubenchmark(
                self.gpu_version,
                gpu_sig,
                up,
                down,
                axis,
            )

            key = self.cpu_version(cpu_sig, up, down, axis)
            assert array_equal(cp.asnumpy(output), key)

    @pytest.mark.benchmark(group="UpFirDn2d")
    @pytest.mark.parametrize("num_samps", [2 ** 8])
    @pytest.mark.parametrize("up", [2, 3, 7])
    @pytest.mark.parametrize("down", [1, 2, 9])
    @pytest.mark.parametrize("axis", [-1, 0])
    class TestUpFirDn2d:
        def cpu_version(self, sig, up, down, axis):
            return signal.upfirdn([1, 1, 1], sig, up, down, axis)

        def gpu_version(self, sig, up, down, axis):
            with cp.cuda.Stream.null:
                out = cusignal.upfirdn([1, 1, 1], sig, up, down, axis)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_upfirdn2d_cpu(
            self, rand_2d_data_gen, benchmark, num_samps, up, down, axis
        ):
            cpu_sig, _ = rand_2d_data_gen(num_samps)
            benchmark(
                self.cpu_version,
                cpu_sig,
                up,
                down,
                axis,
            )

        def test_upfirdn2d_gpu(
            self,
            rand_2d_data_gen,
            gpubenchmark,
            num_samps,
            up,
            down,
            axis,
        ):

            cpu_sig, gpu_sig = rand_2d_data_gen(num_samps)
            output = gpubenchmark(
                self.gpu_version,
                gpu_sig,
                up,
                down,
                axis,
            )

            key = self.cpu_version(cpu_sig, up, down, axis)
            assert array_equal(cp.asnumpy(output), key)

    @pytest.mark.benchmark(group="Firfilter")
    @pytest.mark.parametrize("num_samps", [2 ** 14, 2 ** 18])
    @pytest.mark.parametrize("filter_len", [8, 32, 128])
    class TestFirfilter:
        def cpu_version(self, sig, filt):
            return signal.lfilter(filt, 1, sig)

        def gpu_version(self, sig, filt):
            with cp.cuda.Stream.null:
                out = cusignal.firfilter(filt, sig)
            cp.cuda.Stream.null.synchronize()
            return out

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
                self.gpu_version,
                gpu_sig,
                gpu_filter,
            )

            key = self.cpu_version(cpu_sig, cpu_filter)
            assert array_equal(cp.asnumpy(output), key)
