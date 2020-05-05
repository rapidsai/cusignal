# linear filter
import cupy as cp
import numpy as np

import cusignal
from scipy import signal

num_samps = 10

np.random.seed(1234)

# cpu_sig = np.random.rand(num_samps) / num_samps
cpu_sig = np.arange(num_samps) / num_samps
gpu_sig = cp.asarray(cpu_sig)

# a = [1.2, 0.27, 0.59]
# b = [1.8, 0.35, 0.71]

a = [1.0, 0.25, 0.5]
b = [1.0, 0.0, 0.0]

d_a = cp.asarray(a)
d_b = cp.asarray(b)

cpu_lfilter = signal.lfilter(b, a, cpu_sig)
gpu_lfilter = cp.asnumpy(cusignal.lfilter(d_b, d_a, gpu_sig, ))

print('******************************************')
print(cpu_lfilter)
print()
print(gpu_lfilter)
