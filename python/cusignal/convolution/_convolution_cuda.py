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

from string import Template

from ..utils._caches import _cupy_kernel_cache
from .convolution_utils import (
    FULL,
    SAME,
    VALID,
    CIRCULAR,
    REFLECT,
    PAD,
    _iDivUp,
    _valfrommode,
    _bvalfromboundary,
)


# Custom Cupy raw kernel implementing upsample, filter, downsample operation
# Matthew Nicely - mnicely@nvidia.com
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


class _cupy_convolve_wrapper(object):
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
        self, d_inp, d_kernel, mode, swapped_inputs, out,
    ):

        kernel_args = (
            d_inp,
            d_inp.shape[0],
            d_kernel,
            d_kernel.shape[0],
            mode,
            swapped_inputs,
            out,
            out.shape[0],
        )

        with self.stream:
            self.kernel(self.grid, self.block, kernel_args)


class _cupy_convolve_2d_wrapper(object):
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
        self, d_inp, paddedW, paddedH, d_kernel, S0, S1, out, outW, outH, pick,
    ):

        kernel_args = (
            d_inp,
            paddedW,
            paddedH,
            d_kernel,
            d_kernel.shape[0],
            d_kernel.shape[1],
            S0,
            S1,
            out,
            outW,
            outH,
            pick,
        )

        with self.stream:
            self.kernel(self.grid, self.block, kernel_args)


def _get_backend_kernel(
    dtype, grid, block, stream, k_type,
):
    from ..utils.compile_kernels import GPUKernel

    kernel = _cupy_kernel_cache[(str(dtype), k_type.value)]
    if kernel:
        if k_type == GPUKernel.CONVOLVE or k_type == GPUKernel.CORRELATE:
            return _cupy_convolve_wrapper(grid, block, stream, kernel)
        elif (
            k_type == GPUKernel.CONVOLVE2D
            or k_type == GPUKernel.CORRELATE2D
        ):
            return _cupy_convolve_2d_wrapper(grid, block, stream, kernel)
        else:
            raise NotImplementedError(
                "No CuPY kernel found for k_type {}, datatype {}".format(
                    k_type, dtype
                )
            )
    else:
        raise ValueError(
            "Kernel {} not found in _cupy_kernel_cache".format(k_type)
        )


def _convolve_gpu(
    inp, out, ker, mode, use_convolve, swapped_inputs, cp_stream,
):
    from ..utils.compile_kernels import _populate_kernel_cache, GPUKernel

    d_inp = cp.array(inp)
    d_kernel = cp.array(ker)

    device_id = cp.cuda.Device()
    numSM = device_id.attributes["MultiProcessorCount"]

    threadsperblock = 256
    blockspergrid = numSM * 20

    if use_convolve:
        _populate_kernel_cache(out.dtype, GPUKernel.CONVOLVE)
        kernel = _get_backend_kernel(
            out.dtype,
            blockspergrid,
            threadsperblock,
            cp_stream,
            GPUKernel.CONVOLVE,
        )
    else:
        _populate_kernel_cache(out.dtype, GPUKernel.CORRELATE)
        kernel = _get_backend_kernel(
            out.dtype,
            blockspergrid,
            threadsperblock,
            cp_stream,
            GPUKernel.CORRELATE,
        )

    kernel(d_inp, d_kernel, mode, swapped_inputs, out)

    return out


def _convolve2d_gpu(
    inp,
    out,
    ker,
    mode,
    boundary,
    use_convolve,
    fillvalue,
    cp_stream,
):
    from ..utils.compile_kernels import _populate_kernel_cache, GPUKernel

    if (boundary != PAD) and (boundary != REFLECT) and (boundary != CIRCULAR):
        raise Exception("Invalid boundary flag")

    S = np.zeros(2, dtype=int)

    # If kernel is square and odd
    if ker.shape[0] == ker.shape[1]:  # square
        if ker.shape[0] % 2 == 1:  # odd
            pick = 1
            S[0] = (ker.shape[0] - 1) // 2
            if mode == 2:  # full
                P1 = P2 = P3 = P4 = S[0] * 2
            else:  # same/valid
                P1 = P2 = P3 = P4 = S[0]
        else:  # even
            pick = 2
            S[0] = ker.shape[0] // 2
            if mode == 2:  # full
                P1 = P2 = P3 = P4 = S[0] * 2 - 1
            else:  # same/valid
                if use_convolve:
                    P1 = P2 = P3 = P4 = S[0]
                else:
                    P1 = P3 = S[0] - 1
                    P2 = P4 = S[0]
    else:  # Non-square
        pick = 3
        S[0] = ker.shape[0]
        S[1] = ker.shape[1]
        if mode == 2:  # full
            P1 = S[0] - 1
            P2 = S[0] - 1
            P3 = S[1] - 1
            P4 = S[1] - 1
        else:  # same/valid
            if use_convolve:
                P1 = S[0] // 2
                P2 = S[0] // 2 if (S[0] % 2) else S[0] // 2 - 1
                P3 = S[1] // 2
                P4 = S[1] // 2 if (S[1] % 2) else S[1] // 2 - 1
            else:
                P1 = S[0] // 2 if (S[0] % 2) else S[0] // 2 - 1
                P2 = S[0] // 2
                P3 = S[1] // 2 if (S[1] % 2) else S[1] // 2 - 1
                P4 = S[1] // 2

    if mode == 1:  # SAME
        pad = ((P1, P2), (P3, P4))  # 4x5
        if boundary == REFLECT:
            inp = cp.pad(inp, pad, "symmetric")
        if boundary == CIRCULAR:
            inp = cp.pad(inp, pad, "wrap")
        if boundary == PAD:
            inp = cp.pad(inp, pad, "constant", constant_values=(fillvalue))

    if mode == 2:  # FULL
        pad = ((P1, P2), (P3, P4))
        if boundary == REFLECT:
            inp = cp.pad(inp, pad, "symmetric")
        if boundary == CIRCULAR:
            inp = cp.pad(inp, pad, "wrap")
        if boundary == PAD:
            inp = cp.pad(inp, pad, "constant", constant_values=(fillvalue))

    paddedW = inp.shape[1]
    paddedH = inp.shape[0]

    outW = out.shape[1]
    outH = out.shape[0]

    d_inp = cp.array(inp)
    d_kernel = cp.array(ker)

    threadsperblock = (16, 16)
    blockspergrid = (
        _iDivUp(out.shape[1], threadsperblock[0]),
        _iDivUp(out.shape[0], threadsperblock[1]),
    )

    if use_convolve:
        _populate_kernel_cache(out.dtype, GPUKernel.CONVOLVE2D)
        kernel = _get_backend_kernel(
            out.dtype,
            blockspergrid,
            threadsperblock,
            cp_stream,
            GPUKernel.CONVOLVE2D,
        )
    else:
        _populate_kernel_cache(
            out.dtype, GPUKernel.CORRELATE2D
        )
        kernel = _get_backend_kernel(
            out.dtype,
            blockspergrid,
            threadsperblock,
            cp_stream,
            GPUKernel.CORRELATE2D,
        )

    kernel(
        d_inp, paddedW, paddedH, d_kernel, S[0], S[1], out, outW, outH, pick
    )

    return out


def _convolve(
    in1, in2, use_convolve, swapped_inputs, mode, cp_stream, autosync,
):

    val = _valfrommode(mode)

    # Promote inputs
    promType = cp.promote_types(in1.dtype, in2.dtype)
    in1 = in1.astype(promType)
    in2 = in2.astype(promType)

    # Create empty array to hold number of aout dimensions
    out_dimens = np.empty(in1.ndim, np.int)
    if val == VALID:
        for i in range(in1.ndim):
            out_dimens[i] = (
                max(in1.shape[i], in2.shape[i])
                - min(in1.shape[i], in2.shape[i])
                + 1
            )
            if out_dimens[i] < 0:
                raise Exception(
                    "no part of the output is valid, use option 1 (same) or 2 \
                     (full) for third argument"
                )
    elif val == SAME:
        for i in range(in1.ndim):
            if not swapped_inputs:
                out_dimens[i] = in1.shape[i]  # Per scipy docs
            else:
                out_dimens[i] = min(in1.shape[i], in2.shape[i])
    elif val == FULL:
        for i in range(in1.ndim):
            out_dimens[i] = in1.shape[i] + in2.shape[i] - 1
    else:
        raise Exception("mode must be 0 (valid), 1 (same), or 2 (full)")

    # Create empty array out on GPU
    out = cp.empty(out_dimens.tolist(), in1.dtype)

    out = _convolve_gpu(
        in1, out, in2, val, use_convolve, swapped_inputs, cp_stream,
    )

    if autosync is True:
        cp_stream.synchronize()

    return out


def _convolve2d(
    in1,
    in2,
    use_convolve,
    mode,
    boundary,
    fillvalue,
    cp_stream,
    autosync,
):

    val = _valfrommode(mode)
    bval = _bvalfromboundary(boundary)

    # Promote inputs
    promType = cp.promote_types(in1.dtype, in2.dtype)
    in1 = in1.astype(promType)
    in2 = in2.astype(promType)

    if (bval != PAD) and (bval != REFLECT) and (bval != CIRCULAR):
        raise Exception("Incorrect boundary value.")

    if (bval == PAD) and (fillvalue is not None):
        fill = np.array(fillvalue, in1.dtype)
        if fill is None:
            raise Exception("fill must no be None.")
        if fill.size != 1:
            if fill.size == 0:
                raise Exception("`fillvalue` cannot be an empty array.")
            raise Exception(
                "`fillvalue` must be scalar or an array with one element"
            )
    else:
        fill = np.zeros(1, in1.dtype)
        if fill is None:
            raise Exception("Unable to create fill array")

    # Create empty array to hold number of aout dimensions
    out_dimens = np.empty(in1.ndim, np.int)
    if val == VALID:
        for i in range(in1.ndim):
            out_dimens[i] = in1.shape[i] - in2.shape[i] + 1
            if out_dimens[i] < 0:
                raise Exception(
                    "no part of the output is valid, use option 1 (same) or 2 \
                     (full) for third argument"
                )
    elif val == SAME:
        for i in range(in1.ndim):
            out_dimens[i] = in1.shape[i]
    elif val == FULL:
        for i in range(in1.ndim):
            out_dimens[i] = in1.shape[i] + in2.shape[i] - 1
    else:
        raise Exception("mode must be 0 (valid), 1 (same), or 2 (full)")

    # Create empty array out on GPU
    out = cp.empty(out_dimens.tolist(), in1.dtype)

    out = _convolve2d_gpu(
        in1, out, in2, val, bval, use_convolve, fill, cp_stream,
    )

    if autosync is True:
        cp_stream.synchronize()

    return out
