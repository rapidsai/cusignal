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
from cusignal.test.utils import array_equal
import cusignal
from scipy import signal


@pytest.mark.benchmark(group="GeneralCosine")
@pytest.mark.parametrize("num_samps", [2 ** 15])
class BenchGeneralCosine:
    def cpu_version(self, num_samps, arr):
        return signal.windows.general_cosine(num_samps, arr, sym=False)

    def bench_general_cosine_cpu(self, benchmark, num_samps):
        HFT90D = [1, 1.942604, 1.340318, 0.440811, 0.043097]

        benchmark(self.cpu_version, num_samps, HFT90D)

    def bench_general_cosine_gpu(self, benchmark, num_samps):
        HFT90D = [1, 1.942604, 1.340318, 0.440811, 0.043097]

        output = benchmark(
            cusignal.windows.general_cosine, num_samps, HFT90D, sym=False
        )

        key = self.cpu_version(num_samps, HFT90D)
        assert array_equal(cp.asnumpy(output), key)


@pytest.mark.benchmark(group="Boxcar")
@pytest.mark.parametrize("num_samps", [2 ** 15])
class BenchBoxcar:
    def cpu_version(self, num_samps):
        return signal.windows.boxcar(num_samps)

    def bench_boxcar_cpu(self, benchmark, num_samps):
        benchmark(self.cpu_version, num_samps)

    def bench_boxcar_gpu(self, benchmark, num_samps):
        output = benchmark(cusignal.windows.boxcar, num_samps)

        key = self.cpu_version(num_samps)
        assert array_equal(cp.asnumpy(output), key)


@pytest.mark.benchmark(group="Triang")
@pytest.mark.parametrize("num_samps", [2 ** 15])
class BenchTriang:
    def cpu_version(self, num_samps):
        return signal.windows.triang(num_samps)

    def bench_triang_cpu(self, benchmark, num_samps):
        benchmark(self.cpu_version, num_samps)

    def bench_triang_gpu(self, benchmark, num_samps):
        output = benchmark(cusignal.windows.triang, num_samps)

        key = self.cpu_version(num_samps)
        assert array_equal(cp.asnumpy(output), key)


@pytest.mark.benchmark(group="Bohman")
@pytest.mark.parametrize("num_samps", [2 ** 15])
class BenchBohman:
    def cpu_version(self, num_samps):
        return signal.windows.bohman(num_samps)

    def bench_bohman_cpu(self, benchmark, num_samps):
        benchmark(self.cpu_version, num_samps)

    def bench_bohman_gpu(self, benchmark, num_samps):
        output = benchmark(cusignal.windows.bohman, num_samps)

        key = self.cpu_version(num_samps)
        assert array_equal(cp.asnumpy(output), key)


@pytest.mark.benchmark(group="Blackman")
@pytest.mark.parametrize("num_samps", [2 ** 15])
class BenchBlackman:
    def cpu_version(self, num_samps):
        return signal.windows.blackman(num_samps)

    def bench_blackman_cpu(self, benchmark, num_samps):
        benchmark(self.cpu_version, num_samps)

    def bench_blackman_gpu(self, benchmark, num_samps):
        output = benchmark(cusignal.windows.blackman, num_samps)

        key = self.cpu_version(num_samps)
        assert array_equal(cp.asnumpy(output), key)


@pytest.mark.benchmark(group="Nuttall")
@pytest.mark.parametrize("num_samps", [2 ** 15])
class BenchNuttall:
    def cpu_version(self, num_samps):
        return signal.windows.nuttall(num_samps)

    def bench_nuttall_cpu(self, benchmark, num_samps):
        benchmark(self.cpu_version, num_samps)

    def bench_nuttall_gpu(self, benchmark, num_samps):
        output = benchmark(cusignal.windows.nuttall, num_samps)

        key = self.cpu_version(num_samps)
        assert array_equal(cp.asnumpy(output), key)


@pytest.mark.benchmark(group="BlackmanHarris")
@pytest.mark.parametrize("num_samps", [2 ** 15])
class BenchBlackmanHarris:
    def cpu_version(self, num_samps):
        return signal.windows.blackmanharris(num_samps)

    def bench_blackmanharris_cpu(self, benchmark, num_samps):
        benchmark(self.cpu_version, num_samps)

    def bench_blackmanharris_gpu(self, benchmark, num_samps):
        output = benchmark(cusignal.windows.blackmanharris, num_samps)

        key = self.cpu_version(num_samps)
        assert array_equal(cp.asnumpy(output), key)


@pytest.mark.benchmark(group="FlatTop")
@pytest.mark.parametrize("num_samps", [2 ** 15])
class BenchFlatTop:
    def cpu_version(self, num_samps):
        return signal.windows.flattop(num_samps)

    def bench_flattop_cpu(self, benchmark, num_samps):
        benchmark(self.cpu_version, num_samps)

    def bench_flattop_gpu(self, benchmark, num_samps):
        output = benchmark(cusignal.windows.flattop, num_samps)

        key = self.cpu_version(num_samps)
        assert array_equal(cp.asnumpy(output), key)


@pytest.mark.benchmark(group="Barlett")
@pytest.mark.parametrize("num_samps", [2 ** 15])
class BenchBarlett:
    def cpu_version(self, num_samps):
        return signal.windows.bartlett(num_samps)

    def bench_bartlett_cpu(self, benchmark, num_samps):
        benchmark(self.cpu_version, num_samps)

    def bench_bartlett_gpu(self, benchmark, num_samps):
        output = benchmark(cusignal.windows.bartlett, num_samps)

        key = self.cpu_version(num_samps)
        assert array_equal(cp.asnumpy(output), key)


@pytest.mark.benchmark(group="Tukey")
@pytest.mark.parametrize("num_samps", [2 ** 15])
@pytest.mark.parametrize("alpha", [0.25, 0.5])
class BenchTukey:
    def cpu_version(self, num_samps, alpha):
        return signal.windows.tukey(num_samps, alpha, sym=True)

    def bench_tukey_cpu(self, benchmark, num_samps, alpha):
        benchmark(self.cpu_version, num_samps, alpha)

    def bench_tukey_gpu(self, benchmark, num_samps, alpha):
        output = benchmark(cusignal.windows.tukey, num_samps, alpha, sym=True)

        key = self.cpu_version(num_samps, alpha)
        assert array_equal(cp.asnumpy(output), key)


@pytest.mark.benchmark(group="BartHann")
@pytest.mark.parametrize("num_samps", [2 ** 15])
class BenchBartHann:
    def cpu_version(self, num_samps):
        return signal.windows.barthann(num_samps)

    def bench_barthann_cpu(self, benchmark, num_samps):
        benchmark(self.cpu_version, num_samps)

    def bench_barthann_gpu(self, benchmark, num_samps):
        output = benchmark(cusignal.windows.barthann, num_samps)

        key = self.cpu_version(num_samps)
        assert array_equal(cp.asnumpy(output), key)


@pytest.mark.benchmark(group="GeneralHamming")
@pytest.mark.parametrize("num_samps", [2 ** 15])
@pytest.mark.parametrize("alpha", [0.25, 0.5])
class BenchGeneralHamming:
    def cpu_version(self, num_samps, alpha):
        return signal.windows.general_hamming(num_samps, alpha, sym=True)

    def bench_general_hamming_cpu(self, benchmark, num_samps, alpha):
        benchmark(self.cpu_version, num_samps, alpha)

    def bench_general_hamming_gpu(self, benchmark, num_samps, alpha):
        output = benchmark(
            cusignal.windows.general_hamming, num_samps, alpha, sym=True
        )

        key = self.cpu_version(num_samps, alpha)
        assert array_equal(cp.asnumpy(output), key)


@pytest.mark.benchmark(group="Hamming")
@pytest.mark.parametrize("num_samps", [2 ** 15])
class BenchHamming:
    def cpu_version(self, num_samps):
        return signal.windows.hamming(num_samps)

    def bench_hamming_cpu(self, benchmark, num_samps):
        benchmark(self.cpu_version, num_samps)

    def bench_hamming_gpu(self, benchmark, num_samps):
        output = benchmark(cusignal.windows.hamming, num_samps)

        key = self.cpu_version(num_samps)
        assert array_equal(cp.asnumpy(output), key)


@pytest.mark.benchmark(group="Kaiser")
@pytest.mark.parametrize("num_samps", [2 ** 15])
@pytest.mark.parametrize("beta", [0.25, 0.5])
class BenchKaiser:
    def cpu_version(self, num_samps, beta):
        return signal.windows.kaiser(num_samps, beta, sym=True)

    def bench_kaiser_cpu(self, benchmark, num_samps, beta):
        benchmark(self.cpu_version, num_samps, beta)

    def bench_kaiser_gpu(self, benchmark, num_samps, beta):
        output = benchmark(cusignal.windows.kaiser, num_samps, beta, sym=True)

        key = self.cpu_version(num_samps, beta)
        assert array_equal(cp.asnumpy(output), key)


@pytest.mark.benchmark(group="Gaussian")
@pytest.mark.parametrize("num_samps", [2 ** 15])
@pytest.mark.parametrize("std", [3, 7])
class BenchGaussian:
    def cpu_version(self, num_samps, std):
        return signal.windows.gaussian(num_samps, std)

    def bench_gaussian_cpu(self, benchmark, num_samps, std):
        benchmark(self.cpu_version, num_samps, std)

    def bench_gaussian_gpu(self, benchmark, num_samps, std):
        output = benchmark(cusignal.windows.gaussian, num_samps, std)

        key = self.cpu_version(num_samps, std)
        assert array_equal(cp.asnumpy(output), key)


@pytest.mark.benchmark(group="GeneralGaussian")
@pytest.mark.parametrize("num_samps", [2 ** 15])
@pytest.mark.parametrize("p", [0.75, 1.5])
@pytest.mark.parametrize("std", [3, 7])
class BenchGeneralGaussian:
    def cpu_version(self, num_samps, p, std):
        return signal.windows.general_gaussian(num_samps, p, std)

    def bench_general_gaussian_cpu(self, benchmark, num_samps, p, std):
        benchmark(self.cpu_version, num_samps, p, std)

    def bench_general_gaussian_gpu(self, benchmark, num_samps, p, std):
        output = benchmark(
            cusignal.windows.general_gaussian, num_samps, p, std
        )

        key = self.cpu_version(num_samps, p, std)
        assert array_equal(cp.asnumpy(output), key)


@pytest.mark.benchmark(group="Cosine")
@pytest.mark.parametrize("num_samps", [2 ** 15])
class BenchCosine:
    def cpu_version(self, num_samps):
        return signal.windows.cosine(num_samps)

    def bench_cosine_cpu(self, benchmark, num_samps):
        benchmark(self.cpu_version, num_samps)

    def bench_cosine_gpu(self, benchmark, num_samps):
        output = benchmark(cusignal.windows.cosine, num_samps)

        key = self.cpu_version(num_samps)
        assert array_equal(cp.asnumpy(output), key)


@pytest.mark.benchmark(group="Exponential")
@pytest.mark.parametrize("num_samps", [2 ** 15])
@pytest.mark.parametrize("tau", [1.5, 3.0])
class BenchExponential:
    def cpu_version(self, num_samps, tau):
        return signal.windows.exponential(num_samps, tau=tau)

    def bench_exponential_cpu(self, benchmark, num_samps, tau):
        benchmark(self.cpu_version, num_samps, tau)

    def bench_exponential_gpu(self, benchmark, num_samps, tau):
        output = benchmark(cusignal.windows.exponential, num_samps, tau=tau)

        key = self.cpu_version(num_samps, tau)
        assert array_equal(cp.asnumpy(output), key)
