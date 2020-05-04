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

from .. import _filters

class KalmanFilter(object):

    #  documentation
    def __init__(self, num_points, dim_x, dim_z, dim_u=0, use_numba=False):

        self.num_points = num_points
        self.use_numba = use_numba

        if dim_x < 1:
            raise ValueError("dim_x must be 1 or greater")
        if dim_z < 1:
            raise ValueError("dim_z must be 1 or greater")
        if dim_u < 0:
            raise ValueError("dim_u must be 0 or greater")

        if dim_z > 2:
            raise ValueError(
                "cuSignal KalmanFilter only works with dim_z = 2 currently"
            )

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        # 1. if read-only and same initial, we can have one copy
        # 2. if not read-only and same initial, use broadcasting
        self.x = cp.zeros(
            (self.num_points, dim_x, 1,), dtype=cp.float32
        )  # state

        self.P = cp.repeat(
            cp.identity(dim_x, dtype=cp.float32)[cp.newaxis, :, :],
            self.num_points,
            axis=0,
        )  # uncertainty covariance

        self.Q = cp.repeat(
            cp.identity(dim_x, dtype=cp.float32)[cp.newaxis, :, :],
            self.num_points,
            axis=0,
        )  # process uncertainty

        # self.B = None  # control transition matrix

        self.F = cp.repeat(
            cp.identity(dim_x, dtype=cp.float32)[cp.newaxis, :, :],
            self.num_points,
            axis=0,
        )  # state transition matrix

        self.H = cp.zeros(
            (self.num_points, dim_z, dim_z,), dtype=cp.float32
        )  # Measurement function

        self.R = cp.repeat(
            cp.identity(dim_z, dtype=cp.float32)[cp.newaxis, :, :],
            self.num_points,
            axis=0,
        )  # process uncertainty

        self._alpha_sq = cp.ones(
            (self.num_points, 1, 1,), dtype=cp.float32
        )  # fading memory control

        self.M = cp.zeros(
            (self.num_points, dim_z, dim_z,), dtype=cp.float32
        )  # process-measurement cross correlation

        self.z = cp.empty((self.num_points, dim_z, 1,), dtype=cp.float32)

        _filters._populate_kernel_cache(
            self.x.dtype.type, self.use_numba, _filters.GPUKernel.PREDICT
        )
        _filters._populate_kernel_cache(
            self.x.dtype.type, self.use_numba, _filters.GPUKernel.UPDATE
        )

    def predict(self):
        d = cp.cuda.device.Device(0)
        numSM = d.attributes["MultiProcessorCount"]
        threadsperblock = (self.dim_x, self.dim_x, 16)
        blockspergrid = (1, 1, numSM * 20)

        A_size = self.dim_x * self.dim_x
        F_size = self.dim_x * self.dim_x

        total_size = A_size + F_size

        shared_mem_size = (
            total_size * threadsperblock[2] * self.x.dtype.itemsize
        )

        kernel = _filters._get_backend_kernel(
            self.x.dtype,
            blockspergrid,
            threadsperblock,
            shared_mem_size,
            cp.cuda.stream.Stream(null=True),
            self.use_numba,
            _filters.GPUKernel.PREDICT,
        )

        kernel(
            self._alpha_sq, self.x, self.F, self.P, self.Q,
        )

    def update(self):
        d = cp.cuda.device.Device(0)
        numSM = d.attributes["MultiProcessorCount"]
        threadsperblock = (self.dim_x, self.dim_x, 16)
        blockspergrid = (1, 1, numSM * 20)

        A_size = self.dim_x * self.dim_x
        B_size = self.dim_x * self.dim_x
        P_size = self.dim_x * self.dim_x
        H_size = self.dim_z * self.dim_x
        K_size = self.dim_x * self.dim_z
        R_size = self.dim_z * self.dim_z
        y_size = self.dim_z * 1

        total_size = (
            A_size + B_size + P_size + H_size + K_size + R_size + y_size
        )

        shared_mem_size = (
            total_size * threadsperblock[2] * self.x.dtype.itemsize
        )

        kernel = _filters._get_backend_kernel(
            self.x.dtype,
            blockspergrid,
            threadsperblock,
            shared_mem_size,
            cp.cuda.stream.Stream(null=True),
            self.use_numba,
            _filters.GPUKernel.UPDATE,
        )

        kernel(
            self.x, self.z, self.H, self.P, self.R,
        )
