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


class TestPeakFinding:
    @pytest.mark.benchmark(group="Argrelmin")
    @pytest.mark.parametrize("dim, num_samps", [(1, 2**15), (2, 2**8), (3, 2**5)])
    @pytest.mark.parametrize("axis", [-1])
    @pytest.mark.parametrize("order", [1, 2])
    @pytest.mark.parametrize("mode", ["clip", "wrap"])
    class TestArgrelmin:
        def cpu_version(self, sig, axis, order, mode):
            return signal.argrelmin(sig, axis, order, mode)

        def gpu_version(self, sig, axis, order, mode):
            with cp.cuda.Stream.null:
                out = cusignal.argrelmin(sig, axis, order, mode)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_argrelmin_cpu(
            self, rand_data_gen, benchmark, dim, num_samps, axis, order, mode
        ):
            cpu_sig, _ = rand_data_gen(num_samps, dim)
            benchmark(self.cpu_version, cpu_sig, axis, order, mode)

        def test_argrelmin_gpu(
            self,
            rand_data_gen,
            gpubenchmark,
            dim,
            num_samps,
            axis,
            order,
            mode,
        ):
            cpu_sig, gpu_sig = rand_data_gen(num_samps, dim)
            output = gpubenchmark(self.gpu_version, gpu_sig, axis, order, mode)
            key = self.cpu_version(cpu_sig, axis, order, mode)
            array_equal(output, key)

    @pytest.mark.benchmark(group="TestArgrelmax")
    @pytest.mark.parametrize("dim, num_samps", [(1, 2**15), (2, 2**8), (3, 2**5)])
    @pytest.mark.parametrize("axis", [-1])
    @pytest.mark.parametrize("order", [1, 2])
    @pytest.mark.parametrize("mode", ["clip", "wrap"])
    class TestArgrelmax:
        def cpu_version(self, sig, axis, order, mode):
            return signal.argrelmax(sig, axis, order, mode)

        def gpu_version(self, sig, axis, order, mode):
            with cp.cuda.Stream.null:
                out = cusignal.argrelmax(sig, axis, order, mode)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_argrelmax_cpu(
            self, rand_data_gen, benchmark, dim, num_samps, axis, order, mode
        ):
            cpu_sig, _ = rand_data_gen(num_samps, dim)
            benchmark(self.cpu_version, cpu_sig, axis, order, mode)

        def test_argrelmax_gpu(
            self,
            rand_data_gen,
            gpubenchmark,
            dim,
            num_samps,
            axis,
            order,
            mode,
        ):
            cpu_sig, gpu_sig = rand_data_gen(num_samps, dim)
            output = gpubenchmark(self.gpu_version, gpu_sig, axis, order, mode)
            key = self.cpu_version(cpu_sig, axis, order, mode)
            array_equal(output, key)

    @pytest.mark.benchmark(group="Argrelextrema")
    @pytest.mark.parametrize("dim, num_samps", [(1, 2**15), (2, 2**8), (3, 2**5)])
    @pytest.mark.parametrize("axis", [-1])
    @pytest.mark.parametrize("order", [1, 2])
    @pytest.mark.parametrize("mode", ["clip", "wrap"])
    class TestArgrelextrema:
        def cpu_version(self, sig, axis, order, mode):
            return signal.argrelextrema(sig, np.less, axis, order, mode)

        def gpu_version(self, sig, axis, order, mode):
            with cp.cuda.Stream.null:
                out = cusignal.argrelextrema(sig, cp.less, axis, order, mode)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_argrelextrema_cpu(
            self,
            rand_data_gen,
            benchmark,
            dim,
            num_samps,
            axis,
            order,
            mode,
        ):
            cpu_sig, _ = rand_data_gen(num_samps, dim)
            benchmark(self.cpu_version, cpu_sig, axis, order, mode)

        def test_argrelextrema_gpu(
            self,
            rand_data_gen,
            gpubenchmark,
            dim,
            num_samps,
            axis,
            order,
            mode,
        ):
            cpu_sig, gpu_sig = rand_data_gen(num_samps, dim)
            output = gpubenchmark(self.gpu_version, gpu_sig, axis, order, mode)
            key = self.cpu_version(cpu_sig, axis, order, mode)
            array_equal(output, key)
