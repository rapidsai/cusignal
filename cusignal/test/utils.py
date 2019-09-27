import numpy as np
import cupy as cp

def array_equal(a, b, tol=1e-4, with_sign=True):
    res = np.max(np.abs(a-b)) < tol
    return res