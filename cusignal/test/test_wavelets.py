import pytest
import cupy as cp
from cusignal.test.utils import array_equal
import cusignal
import numpy as np
from scipy import signal


@pytest.mark.parametrize('num_samps', [2**14])
def test_morlet(num_samps):
    cpu_window = signal.morlet(num_samps)
    gpu_window = cp.asnumpy(cusignal.morlet(num_samps))
    assert array_equal(cpu_window, gpu_window)


@pytest.mark.parametrize('num_samps', [2**14])
@pytest.mark.parametrize('a', [10, 1000])
def test_ricker(num_samps, a):
    cpu_window = signal.morlet(num_samps)
    gpu_window = cp.asnumpy(cusignal.morlet(num_samps))
    assert array_equal(cpu_window, gpu_window)


@pytest.mark.parametrize('num_samps', [2**14])
@pytest.mark.parametrize('widths', [31, 127])
def test_cwt(num_samps, widths):
    cpu_signal = np.random.rand(int(num_samps))
    gpu_signal = cp.asarray(cpu_signal)

    cpu_cwt = signal.cwt(cpu_signal, signal.ricker, np.arange(1, widths))
    gpu_cwt = cp.asnumpy(
        cusignal.cwt(gpu_signal, cusignal.ricker, cp.arange(1, widths))
    )

    assert array_equal(cpu_cwt, gpu_cwt)


@pytest.mark.parametrize('num_samps', [2**14])
@pytest.mark.parametrize('widths', [31, 127])
def test_cwt_complex(num_samps, widths):
    cpu_signal = (
        np.random.rand(int(num_samps)) + 1j * np.random.rand(int(num_samps))
    )
    gpu_signal = cp.asarray(cpu_signal)

    cpu_cwt = signal.cwt(cpu_signal, signal.ricker, np.arange(1, widths))
    gpu_cwt = cp.asnumpy(
        cusignal.cwt(gpu_signal, cusignal.ricker, cp.arange(1, widths))
    )

    assert array_equal(cpu_cwt, gpu_cwt)
