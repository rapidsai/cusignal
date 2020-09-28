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
from acoustics.cepstrum import complex_cepstrum, real_cepstrum
from cusignal.test.utils import _check_rapids_pytest_benchmark, array_equal
from scipy import signal

gpubenchmark = _check_rapids_pytest_benchmark()

# # Missing
# # cceps_unwrap
# # cceps

def cceps_cpu(x, n=None):
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

def rceps_cpu(x, n=None):
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

class TestAcoustics:
    @pytest.mark.benchmark(group="Rceps")
    @pytest.mark.parametrize("num_samps", [2 ** 8, 2 ** 14])
    class TestRceps:
        def cpu_version(self, sig):
            return rceps_cpu(sig)

        def gpu_version(self, sig):
            with cp.cuda.Stream.null:
                out = cusignal.rceps(sig)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_rceps_cpu(self, rand_data_gen, benchmark, num_samps):
            cpu_sig, _ = rand_data_gen(num_samps)
            benchmark(self.cpu_version, cpu_sig)

        def test_rceps_gpu(self, rand_data_gen, gpubenchmark, num_samps):

            cpu_sig, gpu_sig = rand_data_gen(num_samps)
            output = gpubenchmark(self.gpu_version, gpu_sig)

            key = self.cpu_version(cpu_sig)
            assert array_equal(cp.asnumpy(output), key)


#     @pytest.mark.benchmark(group="CcepsUnwrap")
#     class TestCcepsUnwrap:
#         def cpu_version(self, cpu_sig):
#             return signal.freq_shift(cpu_sig)

#         @pytest.mark.cpu
#         def test_cceps_unwrap_cpu(self, benchmark):
#             benchmark(self.cpu_version, cpu_sig)

#         def test_cceps_unwrap_gpu(self, gpubenchmark):

#             output = gpubenchmark(cusignal.detrend, gpu_sig)

#             key = self.cpu_version(cpu_sig)
#             assert array_equal(cp.asnumpy(output), key)

# @pytest.mark.benchmark(group="Cceps")
# @pytest.mark.parametrize("num_samps", [5])
# class TestCceps:
#     def cpu_version(self, sig):
#         return complex_cepstrum(sig)

#     def gpu_version(self, sig):
#         with cp.cuda.Stream.null:
#             out = cusignal.cceps(sig)
#         cp.cuda.Stream.null.synchronize()
#         return out

#     @pytest.mark.cpu
#     def test_cceps_cpu(self, rand_data_gen, benchmark,  num_samps):
#         cpu_sig, _ = rand_data_gen(num_samps)
#         benchmark(self.cpu_version, cpu_sig)

#     def test_cceps_gpu(self, rand_data_gen, gpubenchmark,  num_samps):

#         cpu_sig, gpu_sig = rand_data_gen(num_samps)
#         output = gpubenchmark(cusignal.detrend, gpu_sig)

#         key = self.cpu_version(cpu_sig)
#         assert array_equal(cp.asnumpy(output), key)
