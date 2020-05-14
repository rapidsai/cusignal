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

import warnings

from numba import cuda, void, float64
from string import Template

from ..utils._caches import _cupy_kernel_cache, _numba_kernel_cache


def _numba_sosfilt(sos, x_in, zi, a, b):

    n_samples = x_in.shape[1]
    n_sections = sos.shape[0]

    s_buffer = cuda.shared.array(shape=0, dtype=float64)

    s_data = s_buffer[: (n_sections * x_in.shape[0])]

    x, y = cuda.grid(2)

    s_data[y * n_sections + x] = 0
    cuda.syncthreads()

    load_size = n_sections - 1
    unload_size = n_samples - load_size

    # Loading phase
    for n in range(load_size):
        # for s in range(n_sections):
        if x == 0:
            x_n = x_in[y, n]  # make a temporary copy
        else:
            x_n = s_data[y * n_sections + x - 1]

        # Use direct II transposed structure:
        temp = b[x, 0] * x_n + zi[y, x, 0]

        # print("1", y, n, x, x_n, temp, zi[y, x, 0], a[x, 0], b[x, 0])

        zi[y, x, 0] = (b[x, 1] * x_n - a[x, 0] * temp + zi[y, x, 1])
        zi[y, x, 1] = (b[x, 2] * x_n - a[x, 1] * temp)

        s_data[y * n_sections + x] = temp

        cuda.syncthreads()

    # Processing phase
    for n in range(load_size, n_samples):

        if x == 0:
            x_n = x_in[y, n]  # make a temporary copy
        else:
            x_n = s_data[y * n_sections + x - 1]

        # Use direct II transposed structure:
        temp = b[x, 0] * x_n + zi[y, x, 0]

        # print("1", y, n, x, x_n, temp, zi[y, x, 0], a[x, 0], b[x, 0])

        zi[y, x, 0] = (b[x, 1] * x_n - a[x, 0] * temp + zi[y, x, 1])
        zi[y, x, 1] = (b[x, 2] * x_n - a[x, 1] * temp)

        if x < load_size:
            s_data[y * n_sections + x] = temp
        if x == load_size:
            x_in[y, n - load_size] = temp

        cuda.syncthreads()

    # Unloading phase
    for n in range(n_sections):
        # retire threads that are less than n
        if x > n:
            x_n = s_data[y * n_sections + x - 1]

            # Use direct II transposed structure:
            temp = b[x, 0] * x_n + zi[y, x, 0]

            zi[y, x, 0] = (b[x, 1] * x_n - a[x, 0] * temp + zi[y, x, 1])
            zi[y, x, 1] = (b[x, 2] * x_n - a[x, 1] * temp)

            if x < load_size:
                s_data[y * n_sections + x] = temp
            if x == load_size:
                x_in[y, n + unload_size] = temp

            cuda.syncthreads()


def _numba_sosfilt_signature(ty):
    return void(
        ty[:, :], ty[:, :], ty[:, :, :], ty[:, :], ty[:, :],
    )


# Custom Cupy raw kernel implementing lombscargle operation
# Matthew Nicely - mnicely@nvidia.com
_cupy_sosfilt_src = Template(
    """
$header

extern "C" {
    __global__ void _cupy_sosfilt( ) {

        const int tx {
            static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
        const int stride { static_cast<int>( blockDim.x * gridDim.x ) };

    }
}
"""
)


class _cupy_lombscargle_wrapper(object):
    def __init__(self, grid, block, stream, kernel):
        if isinstance(grid, int):
            grid = (grid,)
        if isinstance(block, int):
            block = (block,)

        self.grid = grid
        self.block = block
        self.stream = stream
        self.kernel = kernel

    def __call__(
        self, sos, x, zi, a, b
    ):

        kernel_args = (
        )

        self.stream.use()
        self.kernel(self.grid, self.block, kernel_args)


def _get_backend_kernel(dtype, grid, block, smem, stream, use_numba, k_type):
    from ..utils.compile_kernels import _stream_cupy_to_numba

    if not use_numba:
        kernel = _cupy_kernel_cache[(dtype.name, k_type.value)]
        if kernel:
            return _cupy_lombscargle_wrapper(grid, block, stream, kernel)
        else:
            raise ValueError(
                "Kernel {} not found in _cupy_kernel_cache".format(k_type)
            )

    else:
        warnings.warn(
            "Numba kernels will be removed in a later release",
            FutureWarning,
            stacklevel=4,
        )

        nb_stream = _stream_cupy_to_numba(stream)
        kernel = _numba_kernel_cache[(dtype.name, k_type.value)]

        if kernel:
            return kernel[grid, block, nb_stream, smem]
        else:
            raise ValueError(
                "Kernel {} not found in _numba_kernel_cache".format(k_type)
            )

    raise NotImplementedError(
        "No kernel found for datatype {}".format(dtype.name)
    )


def _sosfilt(sos, x, zi, cp_stream, autosync, use_numba):
    from ..utils.compile_kernels import _populate_kernel_cache, GPUKernel

    # device_id = cp.cuda.Device()
    # numSM = device_id.attributes["MultiProcessorCount"]
    # threadsperblock = 256
    # blockspergrid = numSM * 20

    threadsperblock = (sos.shape[0], x.shape[0])
    blockspergrid = (1, 1)

    _populate_kernel_cache(x.dtype.type, use_numba, GPUKernel.SOSFILT)

    shared_mem = threadsperblock[0] * threadsperblock[1] * x.dtype.itemsize
    print(x.dtype.itemsize)

    kernel = _get_backend_kernel(
        x.dtype,
        blockspergrid,
        threadsperblock,
        shared_mem,
        cp_stream,
        use_numba,
        GPUKernel.SOSFILT,
    )

    b = sos[:, :3]
    a = sos[:, 4:]

    kernel(sos, x, zi, a, b)

    if autosync is True:
        cp_stream.synchronize()
