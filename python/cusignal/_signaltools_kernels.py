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


def _numba_correlate_2d(
    inp, inpW, inpH, kernel, S0, S1, out, outW, outH, pick
):

    y, x = cuda.grid(2)

    if pick != 3:  # non-square
        i = cp.int32(x + S0)
    else:
        i = cp.int32(x + S1)
    j = cp.int32(y + S0)

    oPixelPos = (x, y)
    if (x < outH) and (y < outW):

        temp: out.dtype = 0

        if pick == 1:  # odd
            for k in range(cp.int32(-S0), cp.int32(S0 + 1)):
                for l in range(cp.int32(-S0), cp.int32(S0 + 1)):
                    iPixelPos = (cp.int32(i + k), cp.int32(j + l))
                    coefPos = (cp.int32(k + S0), cp.int32(l + S0))
                    temp += inp[iPixelPos] * kernel[coefPos]

        elif pick == 2:  # even
            for k in range(cp.int32(-S0), cp.int32(S0)):
                for l in range(cp.int32(-S0), cp.int32(S0)):
                    iPixelPos = (cp.int32(i + k), cp.int32(j + l))
                    coefPos = (cp.int32(k + S0), cp.int32(l + S0))
                    temp += inp[iPixelPos] * kernel[coefPos]

        else:  # non-squares
            for k in range(cp.int32(S0)):
                for l in range(cp.int32(S1)):
                    iPixelPos = (
                        cp.int32(cp.int32(i + k) - S1),
                        cp.int32(cp.int32(j + l) - S0),
                    )
                    coefPos = (k, l)
                    temp += inp[iPixelPos] * kernel[coefPos]

        out[oPixelPos] = temp


def _numba_convolve_2d(inp, inpW, inpH, kernel, S0, S1, out, outW, outH, pick):

    y, x = cuda.grid(2)

    if pick != 3:  # non-square
        i = cp.int32(x + S0)
    else:
        i = cp.int32(x + S1)
    j = cp.int32(y + S0)

    oPixelPos = (x, y)
    if (x < outH) and (y < outW):

        temp: out.dtype = 0

        if pick == 1:  # odd
            for k in range(cp.int32(-S0), cp.int32(S0 + 1)):
                for l in range(cp.int32(-S0), cp.int32(S0 + 1)):
                    iPixelPos = (cp.int32(i + k), cp.int32(j + l))
                    coefPos = (cp.int32(-k + S0), cp.int32(-l + S0))
                    temp += inp[iPixelPos] * kernel[coefPos]

        elif pick == 2:  # even
            for k in range(cp.int32(-S0), cp.int32(S0)):
                for l in range(cp.int32(-S0), cp.int32(S0)):
                    iPixelPos = (cp.int32(i + k), cp.int32(j + l))
                    coefPos = (
                        cp.int32(cp.int32(-k + S0) - 1),
                        cp.int32(cp.int32(-l + S0) - 1),
                    )
                    temp += inp[iPixelPos] * kernel[coefPos]

        else:  # non-squares
            for k in range(cp.int32(S0)):
                for l in range(cp.int32(S1)):
                    iPixelPos = (
                        cp.int32(cp.int32(i + k) - S1),
                        cp.int32(cp.int32(j + l) - S0),
                    )
                    coefPos = (
                        cp.int32(cp.int32(-k + S0) - 1),
                        cp.int32(cp.int32(-l + S1) - 1),
                    )
                    temp += inp[iPixelPos] * kernel[coefPos]

        out[oPixelPos] = temp


def _numba_convolve_2d_signature(ty):
    return void(
        ty[:, :],  # inp
        int64,  # inpW
        int64,  # inpH
        ty[:, :],  # kernel
        int64,  # S0
        int64,  # S1 - only used by non-squares
        ty[:, :],  # out
        int64,  # outW
        int64,  # outH
        int64,  # pick
    )


# Custom Cupy raw kernel implementing upsample, filter, downsample operation
# Matthew Nicely - mnicely@nvidia.com
_cupy_correlate_src = Template(
    """
$header

extern "C" {
    __global__ void _cupy_correlate(
            const ${datatype} * __restrict__ inp,
            const int inpW,
            const ${datatype} * __restrict__ kernel,
            const int kerW,
            const int mode,
            const bool swapped_inputs,
            ${datatype} * __restrict__ out,
            const int outW) {

        const int tx {
            static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
        const int stride { static_cast<int>( blockDim.x * gridDim.x ) };

        for ( int tid = tx; tid < outW; tid += stride ) {
            ${datatype} temp {};

            if ( mode == 0 ) {  // Valid
                if ( tid >= 0 && tid < inpW ) {
                    for ( int j = 0; j < kerW; j++ ) {
                        temp += inp[tid + j] * kernel[j];
                    }
                }
            } else if ( mode == 1 ) {   // Same
                const int P1 { kerW / 2 };
                int start {};
                if ( !swapped_inputs ) {
                    start = 0 - P1 + tid;
                } else {
                    start = ( ( inpW - 1 ) / 2 ) - ( kerW - 1 ) + tid;
                }
                for ( int j = 0; j < kerW; j++ ) {
                    if ( ( start + j >= 0 ) && ( start + j < inpW ) ) {
                        temp += inp[start + j] * kernel[j];
                    }
                }
            } else {    // Full
                const int P1 { kerW - 1 };
                int start { 0 - P1 + tid };
                for ( int j = 0; j < kerW; j++ ) {
                    if ( ( start + j >= 0 ) && ( start + j < inpW ) ) {
                        temp += inp[start + j] * kernel[j];
                    }
                }
            }

            if (swapped_inputs) {
                out[outW - tid - 1] = temp; // TODO: Move to shared memory
            } else {
                out[tid] = temp;
            }
        }
    }
}
"""
)

_cupy_convolve_src = Template(
    """
$header

extern "C" {
    __global__ void _cupy_convolve(
            const ${datatype} * __restrict__ inp,
            const int inpW,
            const ${datatype} * __restrict__ kernel,
            const int kerW,
            const int mode,
            const bool swapped_inputs,
            ${datatype} * __restrict__ out,
            const int outW) {

        const int tx {
            static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
        const int stride { static_cast<int>( blockDim.x * gridDim.x ) };

        for ( int tid = tx; tid < outW; tid += stride ) {

            ${datatype} temp {};

            if ( mode == 0 ) {  // Valid
                if ( tid >= 0 && tid < inpW ) {
                    for ( int j = 0; j < kerW; j++ ) {
                        temp += inp[tid + j] * kernel[( kerW - 1 ) - j];
                    }
                }
            } else if ( mode == 1 ) {   // Same
                const int P1 { kerW / 2 };
                int start {};
                if ( !swapped_inputs ) {
                    start = 0 - P1 + tid;
                } else {
                    start = ( ( inpW - 1 ) / 2 ) - ( kerW - 1 ) + tid;
                }
                for ( int j = 0; j < kerW; j++ ) {
                    if ( ( start + j >= 0 ) && ( start + j < inpW ) ) {
                        temp += inp[start + j] * kernel[( kerW - 1 ) - j];
                    }
                }
            } else {    // Full
                const int P1 { kerW - 1 };
                int start { 0 - P1 + tid };
                for ( int j = 0; j < kerW; j++ ) {
                    if ( ( start + j >= 0 ) && ( start + j < inpW ) ) {
                        temp += inp[start + j] * kernel[( kerW - 1 ) - j];
                    }
                }
            }

            out[tid] = temp;
        }
    }
}
"""
)

_cupy_correlate_2d_src = Template(
    """
$header

extern "C" {
    __global__ void _cupy_correlate_2d(
            const ${datatype} * __restrict__ inp,
            const int inpW,
            const int inpH,
            const ${datatype} * __restrict__ kernel,
            const int kerW,
            const int kerH,
            const int S0,
            const int S1,
            ${datatype} * __restrict__ out,
            const int outW,
            const int outH,
            const int pick) {

        const int ty {
            static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) };
        const int tx {
            static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) };

        int i {};
        if ( pick != 3 ) {
            i = tx + S0;
        } else {
            i = tx + S1;
        }
        int j { ty + S0 };

        int2 oPixelPos {tx, ty};
        if ( (tx < outH) && (ty < outW) ) {
            ${datatype} temp {};

            // Odd
            if ( pick == 1) {
                for (int k = -S0; k < (S0 + 1); k++){
                    for (int l = -S0; l < (S0 + 1); l++) {
                        int2 iPixelPos {(i + k), (j + l)};
                        int2 coefPos {(k + S0), (l + S0)};
                        temp += inp[iPixelPos.x * inpW + iPixelPos.y] *
                            kernel[coefPos.x * kerW + coefPos.y];
                    }
                }

            // Even
            } else if (pick == 2) {
                for (int k = -S0; k < S0; k++){
                    for (int l = -S0; l < S0; l++) {
                        int2 iPixelPos {(i + k), (j + l)}; // iPixelPos[1], [0]
                        int2 coefPos {(k + S0), (l + S0)};
                        temp += inp[iPixelPos.x * inpW + iPixelPos.y] *
                            kernel[coefPos.x * kerW + coefPos.y];
                    }
                }

            // Non-squares
            } else {
                for (int k = 0; k < S0; k++){
                    for (int l = 0; l < S1; l++) {
                        int2 iPixelPos {(i + k - S1), (j + l - S0)};
                        int2 coefPos {k, l};
                        temp += inp[iPixelPos.x * inpW + iPixelPos.y] *
                            kernel[coefPos.x * kerH + coefPos.y];
                    }
                }
            }
            out[oPixelPos.x * outW + oPixelPos.y] = temp;
        }
    }
}
"""
)

_cupy_convolve_2d_src = Template(
    """
$header

extern "C" {
    __global__ void _cupy_convolve_2d(
            const ${datatype} * __restrict__ inp,
            const int inpW,
            const int inpH,
            const ${datatype} * __restrict__ kernel,
            const int kerW,
            const int kerH,
            const int S0,
            const int S1,
            ${datatype} * __restrict__ out,
            const int outW,
            const int outH,
            const int pick) {

        const int ty {
            static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) };
        const int tx {
            static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) };

        int i {};
        if ( pick != 3 ) {
            i = tx + S0;
        } else {
            i = tx + S1;
        }
        int j { ty + S0 };

        int2 oPixelPos {tx, ty};
        if ( (tx < outH) && (ty < outW) ) {
            ${datatype} temp {};

            // Odd kernel
            if ( pick == 1) {
                for (int k = -S0; k < (S0 + 1); k++){
                    for (int l = -S0; l < (S0 + 1); l++) {
                        int2 iPixelPos {(i + k), (j + l)};
                        int2 coefPos {(-k + S0), (-l + S0)};
                        temp += inp[iPixelPos.x * inpW + iPixelPos.y] *
                            kernel[coefPos.x * kerW + coefPos.y];
                    }
                }
            // Even kernel
            } else if (pick == 2) {
                for (int k = -S0; k < S0; k++){
                    for (int l = -S0; l < S0; l++) {
                        int2 iPixelPos {(i + k), (j + l)};
                        int2 coefPos {(-k + S0 - 1), (-l + S0 - 1)};
                        temp += inp[iPixelPos.x * inpW + iPixelPos.y] *
                            kernel[coefPos.x * kerW + coefPos.y];
                    }
                }

            // Non-squares kernel
            } else {
                for (int k = 0; k < S0; k++){
                    for (int l = 0; l < S1; l++) {
                        int2 iPixelPos {(i + k - S1), (j + l - S0)};
                        int2 coefPos {(-k + S0 - 1), (-l + S1 - 1)};
                        temp += inp[iPixelPos.x * inpW + iPixelPos.y] *
                            kernel[coefPos.x * kerH + coefPos.y];
                    }
                }
            }
            out[oPixelPos.x * outW + oPixelPos.y] = temp;
        }
    }
}
"""
)
