# Copyright (c) 2019-2020, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# 1
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cupy as cp
import pkg_resources

from . import _filters
from ..utils.debugtools import print_atts


class KalmanFilter(object):

    # Check CuPy version
    ver = pkg_resources.get_distribution("cupy").version
    if ver != "8.0.0b4" or ver != "8.0.0rc1" or ver != "8.0.0":
        pass
    else:
        raise NotImplementedError(
            "Kalman Filter only compatible with CuPy v.8.0.0b4+"
        )

    #  documentation
    def __init__(
        self, num_points, dim_x, dim_z, dim_u=0, dtype=cp.float32,
    ):

        self.num_points = num_points

        if dim_x < 1:
            raise ValueError("dim_x must be 1 or greater")
        if dim_z < 1:
            raise ValueError("dim_z must be 1 or greater")
        if dim_u < 0:
            raise ValueError("dim_u must be 0 or greater")

        if dim_z > 4:
            raise ValueError(
                "cuSignal KalmanFilter only works with dim_z = 2 currently"
            )

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        # Create data arrays
        self.x = cp.zeros((self.num_points, dim_x, 1,), dtype=dtype)  # state

        self.P = cp.repeat(
            cp.identity(dim_x, dtype=dtype)[cp.newaxis, :, :],
            self.num_points,
            axis=0,
        )  # uncertainty covariance

        self.Q = cp.repeat(
            cp.identity(dim_x, dtype=dtype)[cp.newaxis, :, :],
            self.num_points,
            axis=0,
        )  # process uncertainty

        # self.B = None  # control transition matrix

        self.F = cp.repeat(
            cp.identity(dim_x, dtype=dtype)[cp.newaxis, :, :],
            self.num_points,
            axis=0,
        )  # state transition matrix

        self.H = cp.zeros(
            (self.num_points, dim_z, dim_z,), dtype=dtype
        )  # Measurement function

        self.R = cp.repeat(
            cp.identity(dim_z, dtype=dtype)[cp.newaxis, :, :],
            self.num_points,
            axis=0,
        )  # process uncertainty

        self._alpha_sq = cp.ones(
            (self.num_points, 1, 1,), dtype=dtype
        )  # fading memory control

        self.M = cp.zeros(
            (self.num_points, dim_z, dim_z,), dtype=dtype
        )  # process-measurement cross correlation

        self.z = cp.empty((self.num_points, dim_z, 1,), dtype=dtype)

        # Allocate GPU resources
        threads_z_axis = 16
        d = cp.cuda.device.Device()
        numSM = d.attributes["MultiProcessorCount"]
        threadsperblock = (self.dim_x, self.dim_x, threads_z_axis)
        blockspergrid = (1, 1, numSM * 20)

        max_threads_per_block = self.dim_x * self.dim_x * threads_z_axis

        # Only need to populate cache once
        # At class initialization
        _filters._populate_kernel_cache(
            self.x.dtype,
            threads_z_axis,
            self.dim_x,
            self.dim_z,
            max_threads_per_block,
        )

        # Retrieve kernel from cache
        self.predict_kernel = _filters._get_backend_kernel(
            self.x.dtype,
            blockspergrid,
            threadsperblock,
            _filters.GPUKernel.PREDICT,
        )

        self.update_kernel = _filters._get_backend_kernel(
            self.x.dtype,
            blockspergrid,
            threadsperblock,
            _filters.GPUKernel.UPDATE,
        )

        print_atts(self.predict_kernel)
        print_atts(self.predict_kernel)

    def predict(self):

        self.predict_kernel(
            self._alpha_sq, self.x, self.F, self.P, self.Q,
        )

    def update(self):

        self.update_kernel(
            self.x, self.z, self.H, self.P, self.R,
        )
