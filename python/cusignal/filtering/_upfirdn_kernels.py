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

from numba import cuda, int64, void
from string import Template


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
