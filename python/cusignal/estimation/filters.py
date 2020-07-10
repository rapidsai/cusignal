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

from . import _filters


class KalmanFilter(object):

    #  documentation
    def __init__(
        self,
        num_points,
        dim_x,
        dim_z,
        dim_u=0,
        dtype=cp.float32,
        cp_stream=cp.cuda.stream.Stream.null,
        use_numba=False,
    ):

        self.num_points = num_points
        self.cp_stream = cp_stream
        self.use_numba = use_numba

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
        d = cp.cuda.device.Device(0)
        numSM = d.attributes["MultiProcessorCount"]
        self.threadsperblock = (self.dim_x, self.dim_x, threads_z_axis)
        self.blockspergrid = (1, 1, numSM * 20)

        max_available_threadsperblock = d.attributes[
            "MaxThreadsPerMultiProcessor"
        ]

        max_threads_per_block = self.dim_x * self.dim_x * threads_z_axis
        min_blocks_per_multiprocessor = (
            max_available_threadsperblock // max_threads_per_block
        )

        # Only need to populate cache once
        # At class initialization
        _filters._populate_kernel_cache(
            self.x.dtype.type,
            self.use_numba,
            self.dim_x,
            self.dim_z,
            max_threads_per_block,
            min_blocks_per_multiprocessor,
        )

        # #  Only need this for Numba
        # A_size = self.dim_x * self.dim_x
        # F_size = self.dim_x * self.dim_x
        # B_size = self.dim_x * self.dim_x
        # P_size = self.dim_x * self.dim_x
        # H_size = self.dim_z * self.dim_x
        # K_size = self.dim_x * self.dim_z
        # R_size = self.dim_z * self.dim_z
        # y_size = self.dim_z * 1

        # predict_size = A_size + F_size

        # update_size = (
        #     A_size + B_size + P_size + H_size + K_size + R_size + y_size
        # )

        # self.predict_sem = (
        #     predict_size * self.threadsperblock[2] * self.x.dtype.itemsize
        # )
        # self.update_sem = (
        #     update_size * self.threadsperblock[2] * self.x.dtype.itemsize
        # )

        # Retrieve kernel from cache
        self.predict_kernel = _filters._get_backend_kernel(
            self.x.dtype,
            self.blockspergrid,
            self.threadsperblock,
            self.cp_stream,
            self.use_numba,
            _filters.GPUKernel.PREDICT,
        )

        self.update_kernel = _filters._get_backend_kernel(
            self.x.dtype,
            self.blockspergrid,
            self.threadsperblock,
            self.cp_stream,
            self.use_numba,
            _filters.GPUKernel.UPDATE,
        )

        # debug
        # if use_numba is False:
        #     print("Predict")
        #     print(self.predict_kernel.kernel.const_size_bytes)
        #     print(self.predict_kernel.kernel.local_size_bytes)
        #     print(self.predict_kernel.kernel.max_dynamic_shared_size_bytes)
        #     print(self.predict_kernel.kernel.max_threads_per_block)
        #     print(self.predict_kernel.kernel.num_regs)
        #     print(self.predict_kernel.kernel.shared_size_bytes)
        #     print()
        #     print("Update")
        #     print(self.update_kernel.kernel.const_size_bytes)
        #     print(self.update_kernel.kernel.local_size_bytes)
        #     print(self.update_kernel.kernel.max_dynamic_shared_size_bytes)
        #     print(self.update_kernel.kernel.max_threads_per_block)
        #     print(self.update_kernel.kernel.num_regs)
        #     print(self.update_kernel.kernel.shared_size_bytes)
        #     print()
        #     print(max_threads_per_block)
        #     print(min_blocks_per_multiprocessor)
        #     print()

    def predict(self):

        self.predict_kernel(
            self._alpha_sq, self.x, self.F, self.P, self.Q,
        )

    def update(self):

        self.update_kernel(
            self.x, self.z, self.H, self.P, self.R,
        )
