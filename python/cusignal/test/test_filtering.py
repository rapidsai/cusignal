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
import pytest_benchmark

from cusignal.test.utils import array_equal
from scipy import signal

cusignal.precompile_kernels()

try:
    from rapids_pytest_benchmark import setFixtureParamNames
except ImportError:
    print(
        "\n\nWARNING: rapids_pytest_benchmark is not installed, "
        "falling back to pytest_benchmark fixtures.\n"
    )

    # if rapids_pytest_benchmark is not available, just perfrom time-only
    # benchmarking and replace the util functions with nops
    gpubenchmark = pytest_benchmark.plugin.benchmark

    def setFixtureParamNames(*args, **kwargs):
        pass


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
    # class TestLfiltic:
    #     def cpu_version(self, cpu_sig):
    #         return signal.lfiltic(cpu_sig)

    #         @pytest.mark.cpu
    #     def test_lfiltic_cpu(self, benchmark):
    #         benchmark(self.cpu_version, cpu_sig)

    #     def test_lfiltic_gpu(self, gpubenchmark):

    #         output = gpubenchmark(cusignal.lfiltic, gpu_sig)

    #         key = self.cpu_version(cpu_sig)
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
    # class TestDetrend:
    #     def cpu_version(self, cpu_sig):
    #         return signal.detrend(cpu_sig)

    #         @pytest.mark.cpu
    #     def test_detrend_cpu(self, benchmark):
    #         benchmark(self.cpu_version, cpu_sig)

    #     def test_detrend_gpu(self, gpubenchmark):

    #         output = gpubenchmark(cusignal.detrend, gpu_sig)

    #         key = self.cpu_version(cpu_sig)
    #         assert array_equal(cp.asnumpy(output), key)

    # @pytest.mark.benchmark(group="FreqShift")
    # class TestFreqShift:
    #     def cpu_version(self, cpu_sig):
    #         return signal.freq_shift(cpu_sig)

    #         @pytest.mark.cpu
    #     def test_freq_shift_cpu(self, benchmark):
    #         benchmark(self.cpu_version, cpu_sig)

    #     def test_freq_shift_gpu(self, gpubenchmark):

    #         output = gpubenchmark(cusignal.detrend, gpu_sig)

    #         key = self.cpu_version(cpu_sig)
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
