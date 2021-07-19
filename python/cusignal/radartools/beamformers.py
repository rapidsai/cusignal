# Copyright (c) 2020-2021, NVIDIA CORPORATION.
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


def mvdr(x, sv):
    """
    Minimum variance distortionless response (MVDR) beamformer weights

    Parameters
    ----------
    x : ndarray
        Received signal, assume 2D array with size [num_sensors, num_samples]

    sv: ndarray
        Steering vector, assume 1D array with size [num_sensors, 1]

    Note: Unlike MATLAB where input matrix x is of size MxN where N represents
    the number of array elements, we assume row-major formatted data where each
    row is assumed to be complex-valued data from a given sensor (i.e. NxM)
    """
    if x.shape[0] > x.shape[1]:
        raise ValueError('Matrix has more sensors than samples. Consider \
            transposing and remember cuSignal is row-major, unlike MATLAB')

    if x.shape[0] != sv.shape[0]:
        raise ValueError('Steering Vector and input data do not align')

    R = cp.cov(x)
    R_inv = cp.linalg.inv(R)
    svh = cp.transpose(cp.conj(sv))

    wB = cp.matmul(R_inv, sv)
    # wA is a 1x1 scalar
    wA = cp.matmul(svh, wB)
    w = _wB / wA

    return w
