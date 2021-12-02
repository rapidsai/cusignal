# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

def fm_demod(x, axis=-1):
    """
    Demodulate Frequency Modulated Signal

    Parameters
    ----------
    x : ndarray
        Received complex valued signal or batch of signals

    Returns
    -------
    y : ndarray
        The demodulated output with the same shape as `x`.
    """

    x = cp.asarray(x)
    x_angle = cp.unwrap(cp.angle(x), axis=axis)
    y = cp.diff(x_angle, axis=axis)
    return y
