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
import scipy

from cusignal.test.utils import array_equal, _check_rapids_pytest_benchmark
from scipy import signal

gpubenchmark = _check_rapids_pytest_benchmark()


def freq_shift_cpu(x, freq, fs):
    """
    Frequency shift signal by freq at fs sample rate
    Parameters
    ----------
    x : array_like, complex valued
        The data to be shifted.
    freq : float
        Shift by this many (Hz)
    fs : float
        Sampling rate of the signal
    domain : string
        freq or time
    """
    x = np.asarray(x)
    return x * np.exp(-1j * 2 * np.pi * freq / fs * np.arange(x.size))


def channelize_poly_cpu(x, h, n_chans):
    """
    Polyphase channelize signal into n channels
    Parameters
    ----------
    x : array_like
        The input data to be channelized
    h : array_like
        The 1-D input filter; will be split into n
        channels of int number of taps
    n_chans : int
        Number of channels for channelizer
    Returns
    ----------
    yy : channelized output matrix
    Notes
    ----------
    Currently only supports simple channelizer where channel
    spacing is equivalent to the number of channels used
    """

    # number of taps in each h_n filter
    n_taps = int(len(h) / n_chans)

    # number of outputs
    n_pts = int(len(x) / n_chans)

    dtype = cp.promote_types(x.dtype, h.dtype)

    # order F if input from MATLAB
    h = np.conj(np.reshape(h.astype(dtype=dtype), (n_taps, n_chans)).T)

    vv = np.empty(n_chans, dtype=dtype)

    if x.dtype == np.float32 or x.dtype == np.complex64:
        yy = np.empty((n_chans, n_pts), dtype=np.complex64)
    elif x.dtype == np.float64 or x.dtype == np.complex128:
        yy = np.empty((n_chans, n_pts), dtype=np.complex128)

    reg = np.zeros((n_chans, n_taps), dtype=dtype)

    # instead of n_chans here, this could be channel separation
    for i, nn in enumerate(range(0, len(x), n_chans)):
        reg[:, 1:n_taps] = reg[:, 0 : (n_taps - 1)]
        reg[:, 0] = np.conj(np.flipud(x[nn : (nn + n_chans)]))
        for mm in range(n_chans):
            vv[mm] = np.dot(reg[mm, :], np.atleast_2d(h[mm, :]).T)

        yy[:, i] = np.conj(scipy.fft.fft(vv))

    return yy


class TestFilter:
    @pytest.mark.benchmark(group="Wiener")
    @pytest.mark.parametrize("dim, num_samps", [(1, 2 ** 15), (2, 2 ** 8)])
    class TestWiener:
        def cpu_version(self, sig):
            return signal.wiener(sig)

        def gpu_version(self, sig):
            with cp.cuda.Stream.null:
                out = cusignal.wiener(sig)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_wiener_cpu(self, rand_data_gen, benchmark, dim, num_samps):
            cpu_sig, _ = rand_data_gen(num_samps, dim)
            benchmark(self.cpu_version, cpu_sig)

        def test_wiener_gpu(self, rand_data_gen, gpubenchmark, dim, num_samps):

            cpu_sig, gpu_sig = rand_data_gen(num_samps, dim)
            output = gpubenchmark(self.gpu_version, gpu_sig)

            key = self.cpu_version(cpu_sig)
            array_equal(output, key)

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
            benchmark,
            num_signals,
            num_samps,
            order,
            dtype,
        ):
            cpu_sos = signal.ellip(order, 0.009, 80, 0.05, output="sos")
            cpu_sos = np.array(cpu_sos, dtype=dtype)
            cpu_sig = np.random.random((num_signals, num_samps))
            cpu_sig = np.array(cpu_sig, dtype=dtype)
            benchmark(self.cpu_version, cpu_sos, cpu_sig)

        def test_sosfilt_gpu(
            self,
            gpubenchmark,
            num_signals,
            num_samps,
            order,
            dtype,
        ):

            cpu_sos = signal.ellip(order, 0.009, 80, 0.05, output="sos")
            cpu_sos = np.array(cpu_sos, dtype=dtype)
            gpu_sos = cp.asarray(cpu_sos)
            cpu_sig = np.random.random((num_signals, num_samps))
            cpu_sig = np.array(cpu_sig, dtype=dtype)
            gpu_sig = cp.asarray(cpu_sig)

            output = gpubenchmark(
                self.gpu_version,
                gpu_sos,
                gpu_sig,
            )

            key = self.cpu_version(cpu_sos, cpu_sig)
            array_equal(output, key)

    @pytest.mark.benchmark(group="Hilbert")
    @pytest.mark.parametrize("dim, num_samps", [(1, 2 ** 15), (2, 2 ** 8)])
    class TestHilbert:
        def cpu_version(self, sig):
            return signal.hilbert(sig)

        def gpu_version(self, sig):
            with cp.cuda.Stream.null:
                out = cusignal.hilbert(sig)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_hilbert_cpu(self, rand_data_gen, benchmark, dim, num_samps):
            cpu_sig, _ = rand_data_gen(num_samps, dim)
            benchmark(self.cpu_version, cpu_sig)

        def test_hilbert_gpu(
            self, rand_data_gen, gpubenchmark, dim, num_samps
        ):

            cpu_sig, gpu_sig = rand_data_gen(num_samps, dim)
            output = gpubenchmark(self.gpu_version, gpu_sig)

            key = self.cpu_version(cpu_sig)
            array_equal(output, key)

    @pytest.mark.benchmark(group="Hilbert2")
    @pytest.mark.parametrize("dim, num_samps", [(2, 2 ** 8)])
    class TestHilbert2:
        def cpu_version(self, sig):
            return signal.hilbert2(sig)

        def gpu_version(self, sig):
            with cp.cuda.Stream.null:
                out = cusignal.hilbert2(sig)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_hilbert2_cpu(self, rand_data_gen, benchmark, dim, num_samps):
            cpu_sig, _ = rand_data_gen(num_samps, dim)
            benchmark(self.cpu_version, cpu_sig)

        def test_hilbert2_gpu(
            self, rand_data_gen, gpubenchmark, dim, num_samps
        ):

            cpu_sig, gpu_sig = rand_data_gen(num_samps, dim)
            output = gpubenchmark(self.gpu_version, gpu_sig)

            key = self.cpu_version(cpu_sig)
            array_equal(output, key)

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
            array_equal(output, key)

    @pytest.mark.benchmark(group="FreqShift")
    @pytest.mark.parametrize("dtype", [np.float64, np.complex128])
    @pytest.mark.parametrize("num_samps", [2 ** 8])
    @pytest.mark.parametrize("freq", np.fft.fftfreq(10, 0.1))
    @pytest.mark.parametrize("fs", [0.3])
    class TestFreqShift:
        def cpu_version(self, freq, fs, num_samps):
            return freq_shift_cpu(freq, fs, num_samps)

        def gpu_version(self, freq, fs, num_samps):
            with cp.cuda.Stream.null:
                out = cusignal.freq_shift(freq, fs, num_samps)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_freq_shift_cpu(
            self, rand_data_gen, benchmark, dtype, num_samps, freq, fs
        ):
            cpu_sig, _ = rand_data_gen(num_samps, 1, dtype)
            benchmark(self.cpu_version, cpu_sig, freq, fs)

        def test_freq_shift_gpu(
            self, rand_data_gen, gpubenchmark, dtype, num_samps, freq, fs
        ):
            cpu_sig, gpu_sig = rand_data_gen(num_samps, 1, dtype)
            output = gpubenchmark(self.gpu_version, gpu_sig, freq, fs)

            key = self.cpu_version(cpu_sig, freq, fs)
            array_equal(output, key)

    @pytest.mark.benchmark(group="Decimate")
    @pytest.mark.parametrize("num_samps", [2 ** 14, 2 ** 18])
    @pytest.mark.parametrize("downsample_factor", [2, 3, 4, 8, 64])
    @pytest.mark.parametrize("zero_phase", [True, False])
    @pytest.mark.parametrize("gpupath", [True, False])
    class TestDecimate:
        def cpu_version(self, sig, downsample_factor, zero_phase):
            return signal.decimate(
                sig, downsample_factor, ftype="fir", zero_phase=zero_phase
            )

        def gpu_version(self, sig, downsample_factor, zero_phase, gpupath):
            with cp.cuda.Stream.null:
                out = cusignal.decimate(
                    sig,
                    downsample_factor,
                    zero_phase=zero_phase,
                    gpupath=gpupath,
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
            gpupath,
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
            gpupath,
        ):
            cpu_sig, gpu_sig = linspace_data_gen(
                0, 10, num_samps, endpoint=False
            )
            output = gpubenchmark(
                self.gpu_version,
                gpu_sig,
                downsample_factor,
                zero_phase,
                gpupath,
            )

            key = self.cpu_version(cpu_sig, downsample_factor, zero_phase)
            array_equal(output, key)

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
            array_equal(cp.asnumpy(output), key, atol=1e-4)

    @pytest.mark.benchmark(group="ResamplePoly")
    @pytest.mark.parametrize("dim, num_samps", [(1, 2 ** 14), (1, 2 ** 18), (2, 2 ** 14), (2, 2 ** 18)])
    @pytest.mark.parametrize("up", [8])
    @pytest.mark.parametrize("down", [1])
    @pytest.mark.parametrize("axis", [-1, 0])
    @pytest.mark.parametrize("window", [("kaiser", 0.5)])
    class TestResamplePoly:
        def cpu_version(self, sig, up, down, axis, window):
            return signal.resample_poly(sig, up, down, axis, window=window)

        def gpu_version(self, sig, up, down, axis, window):
            with cp.cuda.Stream.null:
                out = cusignal.resample_poly(sig, up, down, axis, window=window)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_resample_poly_cpu(
            self, linspace_data_gen, benchmark, dim, num_samps, up, down, axis, window
        ):
            if dim == 1:
                cpu_sig, _ = linspace_data_gen(0, 10, num_samps, endpoint=False)
            else:
                cpu_sig = np.random.rand(dim, num_samps)
            benchmark(
                self.cpu_version,
                cpu_sig,
                up,
                down,
                axis,
                window,
            )

        def test_resample_poly_gpu(
            self,
            linspace_data_gen,
            gpubenchmark,
            dim,
            num_samps,
            up,
            down,
            axis,
            window,
        ):
            if dim == 1:
                cpu_sig, gpu_sig = linspace_data_gen(
                    0, 10, num_samps, endpoint=False
                )
            else:
                cpu_sig = np.random.rand(dim, num_samps)
                gpu_sig = cp.array(cpu_sig)

            output = gpubenchmark(
                self.gpu_version,
                gpu_sig,
                up,
                down,
                axis,
                window,
            )

            key = self.cpu_version(cpu_sig, up, down, axis, window)
            array_equal(output, key)

    @pytest.mark.benchmark(group="UpFirDn")
    @pytest.mark.parametrize("dim, num_samps", [(1, 2 ** 14), (2, 2 ** 8)])
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
            self, rand_data_gen, benchmark, dim, num_samps, up, down, axis
        ):
            cpu_sig, _ = rand_data_gen(num_samps, dim)
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
            dim,
            num_samps,
            up,
            down,
            axis,
        ):

            cpu_sig, gpu_sig = rand_data_gen(num_samps, dim)
            output = gpubenchmark(
                self.gpu_version,
                gpu_sig,
                up,
                down,
                axis,
            )

            key = self.cpu_version(cpu_sig, up, down, axis)
            array_equal(output, key)

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
            array_equal(output, key)

    @pytest.mark.benchmark(group="ChannelizePoly")
    @pytest.mark.parametrize(
        "dtype", [np.float32, np.float64, np.complex64, np.complex128]
    )
    @pytest.mark.parametrize("num_samps", [2 ** 12])
    @pytest.mark.parametrize("filt_samps", [2048])
    @pytest.mark.parametrize("n_chan", [64, 128, 256])
    class TestChannelizePoly:
        def cpu_version(self, x, h, n_chan):
            return channelize_poly_cpu(x, h, n_chan)

        def gpu_version(self, x, h, n_chan):
            with cp.cuda.Stream.null:
                out = cusignal.channelize_poly(x, h, n_chan)
            cp.cuda.Stream.null.synchronize()
            return out

        @pytest.mark.cpu
        def test_channelizepoly_cpu(
            self,
            benchmark,
            rand_data_gen,
            dtype,
            num_samps,
            filt_samps,
            n_chan,
        ):
            cpu_sig, _ = rand_data_gen(num_samps, 1, dtype)
            cpu_filt, _ = rand_data_gen(filt_samps, 1, dtype)

            benchmark(self.cpu_version, cpu_sig, cpu_filt, n_chan)

        def test_channelizepoly_gpu(
            self,
            gpubenchmark,
            rand_data_gen,
            dtype,
            num_samps,
            filt_samps,
            n_chan,
        ):

            cpu_sig, gpu_sig = rand_data_gen(num_samps, 1, dtype)
            cpu_filt, gpu_filt = rand_data_gen(filt_samps, 1, dtype)

            output = gpubenchmark(self.gpu_version, gpu_sig, gpu_filt, n_chan)

            key = self.cpu_version(cpu_sig, cpu_filt, n_chan)
            array_equal(output, key)
