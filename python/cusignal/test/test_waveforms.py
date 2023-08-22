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
import numpy as np
import pytest
from scipy import signal

import cusignal
from cusignal.testing.utils import _check_rapids_pytest_benchmark, array_equal

gpubenchmark = _check_rapids_pytest_benchmark()


class TestWaveforms:
    @pytest.mark.benchmark(group="Sawtooth")
    @pytest.mark.parametrize("num_samps", [2**14])
    @pytest.mark.parametrize("w", [0.25, 0.5])
    class TestSawtooth:
        def cpu_version(self, sig, w):
            return signal.sawtooth(sig, w)

        def gpu_version(self, sig, w):
            with cp.cuda.Stream.null:
                out = cusignal.sawtooth(sig, w)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_sawtooth_cpu(self, time_data_gen, benchmark, num_samps, w):
            cpu_sig, _ = time_data_gen(0, 10, num_samps)
            benchmark(self.cpu_version, cpu_sig, w)

        def test_sawtooth_gpu(self, time_data_gen, gpubenchmark, num_samps, w):

            cpu_sig, gpu_sig = time_data_gen(0, 10, num_samps)
            output = gpubenchmark(self.gpu_version, gpu_sig, w)

            key = self.cpu_version(cpu_sig, w)
            array_equal(output, key)

    @pytest.mark.benchmark(group="Square")
    @pytest.mark.parametrize("num_samps", [2**14])
    @pytest.mark.parametrize("duty", [0.25, 0.5])
    class TestSquare:
        def cpu_version(self, sig, duty):
            return signal.square(sig, duty)

        def gpu_version(self, sig, duty):
            with cp.cuda.Stream.null:
                out = cusignal.square(sig, duty)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_square_cpu(self, time_data_gen, benchmark, num_samps, duty):
            cpu_sig, _ = time_data_gen(0, 10, num_samps)
            benchmark(self.cpu_version, cpu_sig, duty)

        def test_square_gpu(self, time_data_gen, gpubenchmark, num_samps, duty):

            cpu_sig, gpu_sig = time_data_gen(0, 10, num_samps)
            output = gpubenchmark(self.gpu_version, gpu_sig, duty)

            key = self.cpu_version(cpu_sig, duty)
            array_equal(output, key)

    @pytest.mark.benchmark(group="GaussPulse")
    @pytest.mark.parametrize("num_samps", [2**14])
    @pytest.mark.parametrize("fc", [0.75, 5])
    @pytest.mark.parametrize("retquad", [True, False])
    @pytest.mark.parametrize("retenv", [True, False])
    class TestGaussPulse:
        def cpu_version(self, sig, fc, retquad, retenv):
            return signal.gausspulse(sig, fc, retquad=retquad, retenv=retenv)

        def gpu_version(self, sig, fc, retquad, retenv):
            with cp.cuda.Stream.null:
                out = cusignal.gausspulse(sig, fc, retquad=retquad, retenv=retenv)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_gausspulse_cpu(
            self, time_data_gen, benchmark, num_samps, fc, retquad, retenv
        ):
            cpu_sig, _ = time_data_gen(0, 10, num_samps)
            benchmark(self.cpu_version, cpu_sig, fc, retquad, retenv)

        def test_gausspulse_gpu(
            self, time_data_gen, gpubenchmark, num_samps, fc, retquad, retenv
        ):

            cpu_sig, gpu_sig = time_data_gen(0, 10, num_samps)
            output = gpubenchmark(self.gpu_version, gpu_sig, fc, retquad, retenv)

            key = self.cpu_version(cpu_sig, fc, retquad, retenv)
            array_equal(output, key)

    @pytest.mark.benchmark(group="Chirp")
    @pytest.mark.parametrize("dtype", ["real", "complex"])
    @pytest.mark.parametrize("num_samps", [2**14])
    @pytest.mark.parametrize("f0", [6])
    @pytest.mark.parametrize("t1", [1])
    @pytest.mark.parametrize("f1", [10])
    @pytest.mark.parametrize("method", ["lin", "quad", "log", "hyp"])
    class TestChirp:
        def cpu_version(self, sig, f0, t1, f1, method, dtype):
            if method in ["linear", "lin", "li"]:
                if dtype == "real":
                    return signal.chirp(sig, f0, t1, f1, method)
                else:
                    beta = (f1 - f0) / t1
                    phase = 2 * np.pi * (f0 * sig + 0.5 * beta * sig * sig)
                    return np.cos(phase) - 1j * np.cos(phase + np.pi / 2)
            else:
                return signal.chirp(sig, f0, t1, f1, method)

        def gpu_version(self, sig, f0, t1, f1, method, dtype):
            with cp.cuda.Stream.null:
                out = cusignal.chirp(sig, f0, t1, f1, method, type=dtype)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_chirp_cpu(
            self,
            time_data_gen,
            benchmark,
            dtype,
            num_samps,
            f0,
            t1,
            f1,
            method,
        ):
            cpu_sig, _ = time_data_gen(0, 10, num_samps)
            benchmark(self.cpu_version, cpu_sig, f0, t1, f1, method, dtype)

        def test_chirp_gpu(
            self,
            time_data_gen,
            gpubenchmark,
            dtype,
            num_samps,
            f0,
            t1,
            f1,
            method,
        ):

            cpu_sig, gpu_sig = time_data_gen(0, 10, num_samps)
            output = gpubenchmark(self.gpu_version, gpu_sig, f0, t1, f1, method, dtype)

            key = self.cpu_version(cpu_sig, f0, t1, f1, method, dtype)
            array_equal(output, key)

    @pytest.mark.benchmark(group="UnitImpulse")
    @pytest.mark.parametrize("num_samps", [2**14])
    @pytest.mark.parametrize("idx", ["mid"])
    class TestUnitImpulse:
        def cpu_version(self, num_samps, idx):
            return signal.unit_impulse(num_samps, idx)

        def gpu_version(self, num_samps, idx):
            with cp.cuda.Stream.null:
                out = cusignal.unit_impulse(num_samps, idx)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_unit_impulse_cpu(self, benchmark, num_samps, idx):
            benchmark(self.cpu_version, num_samps, idx)

        def test_unit_impulse_gpu(self, gpubenchmark, num_samps, idx):

            output = gpubenchmark(self.gpu_version, num_samps, idx)

            key = self.cpu_version(num_samps, idx)
            array_equal(output, key)
