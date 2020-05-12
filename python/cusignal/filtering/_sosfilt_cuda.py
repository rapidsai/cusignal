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
import numpy as np
import warnings

from numba import cuda, int64, void
from string import Template

from ..utils._caches import _cupy_kernel_cache, _numba_kernel_cache


def _sosfilt(sos, x, zi):

    n_signals, n_samples = x.shape
    n_sections = sos.shape[0]
    b = sos[:, :3]
    a = sos[:, 4:]

    for i in range(n_signals):
        for n in range(n_samples):
            for s in range(n_sections):
                x_n = x[i, n]  # make a temporary copy
                # Use direct II transposed structure:
                x[i, n] = b[s, 0] * x_n + zi[i, s, 0]
                zi[i, s, 0] = (
                    b[s, 1] * x_n - a[s, 0] * x[i, n] + zi[i, s, 1])
                zi[i, s, 1] = (
                    b[s, 2] * x_n - a[s, 1] * x[i, n])
