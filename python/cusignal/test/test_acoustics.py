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

from cusignal.test.utils import _check_rapids_pytest_benchmark, array_equal

gpubenchmark = _check_rapids_pytest_benchmark()


# https://github.com/python-acoustics/python-acoustics/blob/master/acoustics/cepstrum.py
def complex_cepstrum(x, n=None):
    """Compute the complex cepstrum of a real sequence.
    Parameters
    ----------
    x : ndarray
        Real sequence to compute complex cepstrum of.
    n : {None, int}, optional
        Length of the Fourier transform.
    Returns
    -------
    ceps : ndarray
        The complex cepstrum of the real data sequence `x` computed using the
        Fourier transform.
    ndelay : int
        The amount of samples of circular delay added to `x`.
    The complex cepstrum is given by
    .. math:: c[n] = F^{-1}\\left{\\log_{10}{\\left(F{x[n]}\\right)}\\right}
    where :math:`x_[n]` is the input signal and :math:`F` and :math:`F_{-1}
    are respectively the forward and backward Fourier transform.
    --------
    """

    def _unwrap(phase):
        samples = phase.shape[-1]
        unwrapped = np.unwrap(phase)
        center = (samples + 1) // 2
        if samples == 1:
            center = 0
        ndelay = np.array(np.round(unwrapped[..., center] / np.pi))
        unwrapped -= np.pi * ndelay[..., None] * np.arange(samples) / center
        return unwrapped, ndelay

    spectrum = np.fft.fft(x, n=n)
    unwrapped_phase, ndelay = _unwrap(np.angle(spectrum))
    log_spectrum = np.log(np.abs(spectrum)) + 1j * unwrapped_phase
    ceps = np.fft.ifft(log_spectrum).real

    return ceps, ndelay


def real_cepstrum(x, n=None):
    """
    Compute the real cepstrum of a real sequence.
    x : ndarray
        Real sequence to compute real cepstrum of.
    n : {None, int}, optional
        Length of the Fourier transform.
    Returns
    -------
    ceps: ndarray
        The real cepstrum.
    """
    spectrum = np.fft.fft(x, n=n)
    ceps = np.fft.ifft(np.log(np.abs(spectrum))).real

    return ceps


def inverse_complex_cepstrum(ceps, ndelay):
    r"""Compute the inverse complex cepstrum of a real sequence.
    ceps : ndarray
        Real sequence to compute inverse complex cepstrum of.
    ndelay: int
        The amount of samples of circular delay added to `x`.
    Returns
    -------
    x : ndarray
        The inverse complex cepstrum of the real sequence `ceps`.
    The inverse complex cepstrum is given by
    .. math:: x[n] = F^{-1}\left{\exp(F(c[n]))\right}
    where :math:`c_[n]` is the input signal and :math:`F` and :math:`F_{-1}
    are respectively the forward and backward Fourier transform.
    """

    def _wrap(phase, ndelay):
        ndelay = np.array(ndelay)
        samples = phase.shape[-1]
        center = (samples + 1) // 2
        wrapped = (
            phase + np.pi * ndelay[..., None] * np.arange(samples) / center
        )
        return wrapped

    log_spectrum = np.fft.fft(ceps)
    spectrum = np.exp(
        log_spectrum.real + 1j * _wrap(log_spectrum.imag, ndelay)
    )
    x = np.fft.ifft(spectrum).real

    return x


def minimum_phase(x, n=None):
    r"""Compute the minimum phase reconstruction of a real sequence.
    x : ndarray
        Real sequence to compute the minimum phase reconstruction of.
    n : {None, int}, optional
        Length of the Fourier transform.
    Compute the minimum phase reconstruction of a real sequence using the
    real cepstrum.
    Returns
    -------
    m : ndarray
        The minimum phase reconstruction of the real sequence `x`.
    """
    if n is None:
        n = len(x)
    ceps = real_cepstrum(x, n=n)
    odd = n % 2
    window = np.concatenate(
        (
            [1.0],
            2.0 * np.ones((n + odd) // 2 - 1),
            np.ones(1 - odd),
            np.zeros((n + odd) // 2 - 1),
        )
    )

    m = np.fft.ifft(np.exp(np.fft.fft(window * ceps))).real

    return m


class TestAcoustics:
    @pytest.mark.benchmark(group="ComplexCepstrum")
    @pytest.mark.parametrize("num_samps", [2 ** 8, 2 ** 14])
    @pytest.mark.parametrize("n", [123, 256])
    class TestComplexCepstrum:
        def cpu_version(self, sig, n):
            return complex_cepstrum(sig, n)

        def gpu_version(self, sig, n):
            with cp.cuda.Stream.null:
                out = cusignal.complex_cepstrum(sig, n)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_complex_cepstrum_cpu(
            self, rand_data_gen, benchmark, num_samps, n
        ):
            cpu_sig, _ = rand_data_gen(num_samps)
            benchmark(self.cpu_version, cpu_sig, n)

        def test_complex_cepstrum_gpu(
            self, rand_data_gen, gpubenchmark, num_samps, n
        ):

            cpu_sig, gpu_sig = rand_data_gen(num_samps)
            output = gpubenchmark(self.gpu_version, gpu_sig, n)

            key = self.cpu_version(cpu_sig, n)
            array_equal(output, key)

    @pytest.mark.benchmark(group="RealCepstrum")
    @pytest.mark.parametrize("num_samps", [2 ** 8, 2 ** 14])
    @pytest.mark.parametrize("n", [123, 256])
    class TestRealCepstrum:
        def cpu_version(self, sig, n):
            return real_cepstrum(sig, n)

        def gpu_version(self, sig, n):
            with cp.cuda.Stream.null:
                out = cusignal.real_cepstrum(sig, n)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_real_cepstrum_cpu(
            self, rand_data_gen, benchmark, num_samps, n
        ):
            cpu_sig, _ = rand_data_gen(num_samps)
            benchmark(self.cpu_version, cpu_sig, n)

        def test_real_cepstrum_gpu(
            self, rand_data_gen, gpubenchmark, num_samps, n
        ):

            cpu_sig, gpu_sig = rand_data_gen(num_samps)
            output = gpubenchmark(self.gpu_version, gpu_sig, n)

            key = self.cpu_version(cpu_sig, n)
            array_equal(output, key)

    @pytest.mark.benchmark(group="InverseComplexCepstrum")
    @pytest.mark.parametrize("num_samps", [2 ** 10])
    @pytest.mark.parametrize("n", [123, 256])
    class TestInverseComplexCepstrum:
        def cpu_version(self, sig, n):
            return inverse_complex_cepstrum(sig, n)

        def gpu_version(self, sig, n):
            with cp.cuda.Stream.null:
                out = cusignal.inverse_complex_cepstrum(sig, n)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_inverse_complex_cepstrum_cpu(
            self, rand_data_gen, benchmark, num_samps, n
        ):
            cpu_sig, _ = rand_data_gen(num_samps)
            benchmark(self.cpu_version, cpu_sig, n)

        def test_inverse_complex_cepstrum_gpu(
            self, rand_data_gen, gpubenchmark, num_samps, n
        ):

            cpu_sig, gpu_sig = rand_data_gen(num_samps)
            output = gpubenchmark(self.gpu_version, gpu_sig, n)

            key = self.cpu_version(cpu_sig, n)
            array_equal(output, key)

    @pytest.mark.benchmark(group="MinimumPhase")
    @pytest.mark.parametrize("num_samps", [2 ** 8, 2 ** 14])
    @pytest.mark.parametrize("n", [123, 256])
    class TestMinimumPhase:
        def cpu_version(self, sig, n):
            return minimum_phase(sig, n)

        def gpu_version(self, sig, n):
            with cp.cuda.Stream.null:
                out = cusignal.minimum_phase(sig, n)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_minimum_phase_cpu(
            self, rand_data_gen, benchmark, num_samps, n
        ):
            cpu_sig, _ = rand_data_gen(num_samps)
            benchmark(self.cpu_version, cpu_sig, n)

        def test_minimum_phase_gpu(
            self, rand_data_gen, gpubenchmark, num_samps, n
        ):

            cpu_sig, gpu_sig = rand_data_gen(num_samps)
            output = gpubenchmark(self.gpu_version, gpu_sig, n)

            key = self.cpu_version(cpu_sig, n)
            array_equal(output, key)
