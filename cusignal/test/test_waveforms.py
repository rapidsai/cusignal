import pytest
import cupy as cp
from cusignal.test.utils import array_equal
import cusignal
import numpy as np
from scipy import signal


@pytest.mark.parametrize('num_samps', [2**14])
@pytest.mark.parametrize('duty', [0.25, 0.5])
def test_square(num_samps, duty):
    cpu_time = np.linspace(0, 10, num_samps)
    gpu_time = cp.asarray(cpu_time)

    cpu_pwm = signal.square(cpu_time, duty)
    gpu_pwm = cp.asnumpy(cusignal.square(gpu_time, duty))

    assert array_equal(cpu_pwm, gpu_pwm)

@pytest.mark.parametrize('num_samps', [2**14])
@pytest.mark.parametrize('fc', [0.75, 5])
def test_gausspulse(num_samps, fc):
    cpu_time = np.linspace(0, 10, num_samps)
    gpu_time = cp.asarray(cpu_time)

    cpu_pwm = signal.gausspulse(cpu_time, fc, retquad=True, retenv=True)
    gpu_pwm = cp.asnumpy(cusignal.gausspulse(gpu_time, fc, retquad=True, retenv=True))

    assert array_equal(cpu_pwm, gpu_pwm)

@pytest.mark.parametrize('num_samps', [2**14])
@pytest.mark.parametrize('f0', [6])
@pytest.mark.parametrize('t1', [1])
@pytest.mark.parametrize('f1', [10])
@pytest.mark.parametrize('method', ['linear', 'quadratic'])
def test_chirp(num_samps, f0, t1, f1, method):
    cpu_time = np.linspace(0, 10, num_samps)
    gpu_time = cp.asarray(cpu_time)

    cpu_chirp = signal.chirp(cpu_time, f0, t1, f1, method)
    gpu_chirp = cp.asnumpy(cusignal.chirp(gpu_time, f0, t1, f1, method))

    assert array_equal(cpu_chirp, gpu_chirp)

@pytest.mark.parametrize('num_samps', [2**14])
@pytest.mark.parametrize('idx', ['mid'])
def test_unit_impulse(num_samps, idx):
    cpu_uimp = signal.unit_impulse(num_samps, idx)
    gpu_uimp = cp.asnumpy(cusignal.unit_impulse(num_samps, idx))

    assert array_equal(cpu_uimp, gpu_uimp)