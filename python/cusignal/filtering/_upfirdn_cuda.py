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
import warnings

from math import ceil
from numba import cuda, int64, void
from string import Template

from ..utils._caches import _cupy_kernel_cache, _numba_kernel_cache


def _pad_h(h, up):
    """Store coefficients in a transposed, flipped arrangement.
    For example, suppose upRate is 3, and the
    input number of coefficients is 10, represented as h[0], ..., h[9].
    Then the internal buffer will look like this::
       h[9], h[6], h[3], h[0],   // flipped phase 0 coefs
       0,    h[7], h[4], h[1],   // flipped phase 1 coefs (zero-padded)
       0,    h[8], h[5], h[2],   // flipped phase 2 coefs (zero-padded)
    """
    h_padlen = len(h) + (-len(h) % up)
    h_full = cp.zeros(h_padlen, h.dtype)
    h_full[: len(h)] = h
    h_full = h_full.reshape(-1, up).T[:, ::-1].ravel()
    return h_full


def _output_len(len_h, in_len, up, down):
    in_len_copy = in_len + (len_h + (-len_h % up)) // up - 1
    nt = in_len_copy * up
    need = nt // down
    if nt % down > 0:
        need += 1
    return need


# Custom Numba kernel implementing upsample, filter, downsample operation
# Matthew Nicely - mnicely@nvidia.com
def _numba_upfirdn_1d(
    x, h_trans_flip, up, down, axis, x_shape_a, h_per_phase, padded_len, out
):

    X = cuda.grid(1)
    strideX = cuda.gridsize(1)

    for i in range(X, cp.int32(out.shape[0]), strideX):

        x_idx = cp.int32(cp.int32(cp.int32(i * down) // up) % padded_len)
        h_idx = cp.int32(cp.int32(cp.int32(i * down) % up) * h_per_phase)

        x_conv_idx = cp.int32(cp.int32(x_idx - h_per_phase) + 1)
        if x_conv_idx < 0:
            h_idx -= x_conv_idx
            x_conv_idx = 0

        temp: out.dtype = 0

        # If axis = 0, we need to know each column in x.
        for x_c in range(cp.int32(x_conv_idx), cp.int32(x_idx + 1)):
            if x_c < x_shape_a and x_c >= 0:
                temp += x[x_c] * h_trans_flip[h_idx]
            h_idx += 1

        out[i] = temp


def _numba_upfirdn_2d(
    inp, h_trans_flip, up, down, axis, x_shape_a, h_per_phase, padded_len, out
):

    y, x = cuda.grid(2)

    if x < out.shape[1] and y < out.shape[0]:

        if axis == 1:
            x_idx = cp.int32(cp.int32(cp.int32(x * down) // up) % padded_len)
            h_idx = cp.int32(cp.int32(cp.int32(x * down) % up) * h_per_phase)
        else:
            x_idx = cp.int32(cp.int32(cp.int32(y * down) // up) % padded_len)
            h_idx = cp.int32(cp.int32(cp.int32(y * down) % up) * h_per_phase)

        x_conv_idx = cp.int32(cp.int32(x_idx - h_per_phase) + 1)
        if x_conv_idx < 0:
            h_idx -= x_conv_idx
            x_conv_idx = 0

        temp: out.dtype = 0

        # If axis = 0, we need to know each column in x.
        for x_c in range(cp.int32(x_conv_idx), cp.int32(x_idx + 1)):
            if x_c < x_shape_a and x_c >= 0:  # If inside input
                # if multi-dimenstional array
                if axis == 1:  # process columns
                    temp += inp[y, x_c] * h_trans_flip[h_idx]
                else:  # process rows
                    temp += inp[x_c, x] * h_trans_flip[h_idx]

            h_idx += 1

        out[y, x] = temp


def _numba_upfirdn_1d_signature(ty):
    return void(
        ty[:],  # x
        ty[:],  # h_trans_flip
        int64,  # up
        int64,  # down
        int64,  # axis
        int64,  # x_shape_a
        int64,  # h_per_phase
        int64,  # padded_len
        ty[:],  # out
    )


def _numba_upfirdn_2d_signature(ty):
    return void(
        ty[:, :],  # x
        ty[:],  # h_trans_flip
        int64,  # up
        int64,  # down
        int64,  # axis
        int64,  # x_shape_a
        int64,  # h_per_phase
        int64,  # padded_len
        ty[:, :],  # out
    )


# Custom Cupy raw kernel implementing upsample, filter, downsample operation
# Matthew Nicely - mnicely@nvidia.com
_cupy_upfirdn_1d_src = Template(
    """
$header

extern "C" {
    __global__ void _cupy_upfirdn_1d(
            const ${datatype} * __restrict__ inp,
            const ${datatype} * __restrict__ h_trans_flip,
            const int up,
            const int down,
            const int axis,
            const int x_shape_a,
            const int h_per_phase,
            const int padded_len,
            ${datatype} * __restrict__ out,
            const int outW) {

        const int t {
            static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) };
        const int stride { static_cast<int>(blockDim.x * gridDim.x) };

        for ( int tid = t; tid < outW; tid += stride ) {
            int x_idx { static_cast<int>((tid * down) / up) % padded_len };
            int h_idx { (tid * down) % up * h_per_phase };
            int x_conv_idx { x_idx - h_per_phase + 1 };

            if ( x_conv_idx < 0 ) {
                h_idx -= x_conv_idx;
                x_conv_idx = 0;
            }

            ${datatype} temp {};

            for ( int x_c = x_conv_idx; x_c < (x_idx + 1); x_c++ ) {
                if ( x_c < x_shape_a && x_c >= 0 ) {
                    temp += inp[x_c] * h_trans_flip[h_idx];
                }
                h_idx += 1;
            }
            out[tid] = temp;
        }
    }
}
"""
)

_cupy_upfirdn_2d_src = Template(
    """
$header

extern "C" {
    __global__ void _cupy_upfirdn_2d(
            const ${datatype} * __restrict__ inp,
            const int inpH,
            const ${datatype} * __restrict__ h_trans_flip,
            const int up,
            const int down,
            const int axis,
            const int x_shape_a,
            const int h_per_phase,
            const int padded_len,
            ${datatype} * __restrict__ out,
            const int outW,
            const int outH) {


        const int ty {
            static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) };
        const int tx {
            static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) };

        if ( (tx < outH) && (ty < outW) ) {
            int x_idx {};
            int h_idx {};

            if ( axis == 1 ) {
                x_idx = ( static_cast<int>(tx * down) / up ) % padded_len;
                h_idx = (tx * down) % up * h_per_phase;
            } else {
                x_idx = ( static_cast<int>(ty * down) / up ) % padded_len;
                h_idx = (ty * down) % up * h_per_phase;
            }

            int x_conv_idx { x_idx - h_per_phase + 1 };
            if ( x_conv_idx < 0 ) {
                h_idx -= x_conv_idx;
                x_conv_idx = 0;
            }

            ${datatype} temp {};

            for ( int x_c = x_conv_idx; x_c < (x_idx + 1); x_c++ ) {
                if ( x_c < x_shape_a && x_c >= 0 ) {
                    if (axis == 1) {
                        temp += inp[ty * inpH + x_c] * h_trans_flip[h_idx];
                    } else {
                        temp += inp[x_c * inpH + tx] * h_trans_flip[h_idx];
                    }
                }
                h_idx += 1;
            }
            out[ty * outH + tx] = temp;
        }
    }
}
"""
)


class _cupy_upfirdn_wrapper(object):
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
        self,
        x,
        h_trans_flip,
        up,
        down,
        axis,
        x_shape_a,
        h_per_phase,
        padded_len,
        out,
    ):

        kernel_args = (
            x,
            h_trans_flip,
            up,
            down,
            axis,
            x_shape_a,
            h_per_phase,
            padded_len,
            out,
            out.shape[0],
        )

        with self.stream:
            self.kernel(self.grid, self.block, kernel_args)


class _cupy_upfirdn2d_wrapper(object):
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
        self,
        x,
        h_trans_flip,
        up,
        down,
        axis,
        x_shape_a,
        h_per_phase,
        padded_len,
        out,
    ):

        kernel_args = (
            x,
            x.shape[1],
            h_trans_flip,
            up,
            down,
            axis,
            x_shape_a,
            h_per_phase,
            padded_len,
            out,
            out.shape[0],
            out.shape[1],
        )

        with self.stream:
            self.kernel(self.grid, self.block, kernel_args)


def _get_backend_kernel(
    dtype, grid, block, stream, use_numba, k_type,
):
    from ..utils.compile_kernels import _stream_cupy_to_numba, GPUKernel

    if not use_numba:
        kernel = _cupy_kernel_cache[(dtype.name, k_type.value)]
        if kernel:
            if k_type == GPUKernel.UPFIRDN:
                return _cupy_upfirdn_wrapper(grid, block, stream, kernel)
            elif k_type == GPUKernel.UPFIRDN2D:
                return _cupy_upfirdn2d_wrapper(grid, block, stream, kernel)
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
            return kernel[grid, block, nb_stream]
        else:
            raise ValueError(
                "Kernel {} not found in _numba_kernel_cache".format(k_type)
            )

    raise NotImplementedError(
        "No kernel found for k_type {}, datatype {}".format(k_type, dtype.name)
    )


class _UpFIRDn(object):
    def __init__(self, h, x_dtype, up, down):
        """Helper for resampling"""
        h = cp.asarray(h)
        if h.ndim != 1 or h.size == 0:
            raise ValueError("h must be 1D with non-zero length")

        self._output_type = cp.result_type(h.dtype, x_dtype, cp.float32)
        h = cp.asarray(h, self._output_type)
        self._up = int(up)
        self._down = int(down)
        if self._up < 1 or self._down < 1:
            raise ValueError("Both up and down must be >= 1")
        # This both transposes, and "flips" each phase for filtering
        self._h_trans_flip = _pad_h(h, self._up)
        self._h_trans_flip = cp.ascontiguousarray(self._h_trans_flip)

    def apply_filter(
        self, x, axis, cp_stream, autosync, use_numba,
    ):
        """Apply the prepared filter to the specified axis of a nD signal x"""
        from ..utils.compile_kernels import _populate_kernel_cache, GPUKernel

        output_len = _output_len(
            len(self._h_trans_flip), x.shape[axis], self._up, self._down
        )
        output_shape = cp.asarray(x.shape)
        output_shape[axis] = output_len
        out = cp.zeros(
            cp.asnumpy(output_shape), dtype=self._output_type, order="C"
        )
        axis = axis % x.ndim

        # Precompute variables on CPU
        x_shape_a = x.shape[axis]
        h_per_phase = len(self._h_trans_flip) // self._up
        padded_len = x.shape[axis] + (len(self._h_trans_flip) // self._up) - 1

        if out.ndim > 1:
            threadsperblock = (16, 16)
            blockspergrid_x = ceil(out.shape[0] / threadsperblock[0])
            blockspergrid_y = ceil(out.shape[1] / threadsperblock[1])
            blockspergrid = (blockspergrid_x, blockspergrid_y)

        else:
            device_id = cp.cuda.Device()
            numSM = device_id.attributes["MultiProcessorCount"]
            blockspergrid = numSM * 20
            threadsperblock = 512

        if out.ndim == 1:
            _populate_kernel_cache(
                out.dtype.type, use_numba, GPUKernel.UPFIRDN
            )
            kernel = _get_backend_kernel(
                out.dtype,
                blockspergrid,
                threadsperblock,
                cp_stream,
                use_numba,
                GPUKernel.UPFIRDN,
            )
        elif out.ndim == 2:
            _populate_kernel_cache(
                out.dtype.type, use_numba, GPUKernel.UPFIRDN2D
            )
            kernel = _get_backend_kernel(
                out.dtype,
                blockspergrid,
                threadsperblock,
                cp_stream,
                use_numba,
                GPUKernel.UPFIRDN2D,
            )
        else:
            raise NotImplementedError("upfirdn() requires ndim <= 2")

        kernel(
            cp.asarray(x, self._output_type),
            self._h_trans_flip,
            self._up,
            self._down,
            axis,
            x_shape_a,
            h_per_phase,
            padded_len,
            out,
        )

        if autosync is True:
            cp_stream.synchronize()

        return out
