/*
 * Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <thrust/complex.h>

///////////////////////////////////////////////////////////////////////////////
//                              UPFIRDN1D                                    //
///////////////////////////////////////////////////////////////////////////////

template<typename T>
__device__ void _cupy_upfirdn1D( const T *__restrict__ inp,
                                 const T *__restrict__ h_trans_flip,
                                 const int up,
                                 const int down,
                                 const int axis,
                                 const int x_shape_a,
                                 const int h_per_phase,
                                 const int padded_len,
                                 T *__restrict__ out,
                                 const int outW ) {

    const int t { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
    const int stride { static_cast<int>( blockDim.x * gridDim.x ) };

    for ( size_t tid = t; tid < outW; tid += stride ) {

#if ( __CUDACC_VER_MAJOR__ >= 11 ) && ( __CUDACC_VER_MINOR__ >= 2 )
        __builtin_assume( padded_len > 0 );
        __builtin_assume( up > 0 );
        __builtin_assume( down > 0 );
        __builtin_assume( tid > 0 );
#endif

        const int x_idx { static_cast<int>( ( tid * down ) / up ) % padded_len };
        int       h_idx { static_cast<int>( ( tid * down ) % up * h_per_phase ) };
        int       x_conv_idx { x_idx - h_per_phase + 1 };

        if ( x_conv_idx < 0 ) {
            h_idx -= x_conv_idx;
            x_conv_idx = 0;
        }

        T temp {};

        int stop = ( x_shape_a < ( x_idx + 1 ) ) ? x_shape_a : ( x_idx + 1 );

        for ( int x_c = x_conv_idx; x_c < stop; x_c++ ) {
            temp += inp[x_c] * h_trans_flip[h_idx];
            h_idx += 1;
        }
        out[tid] = temp;
    }
}

extern "C" __global__ void __launch_bounds__( 512 ) _cupy_upfirdn1D_float32( const float *__restrict__ inp,
                                                                             const float *__restrict__ h_trans_flip,
                                                                             const int up,
                                                                             const int down,
                                                                             const int axis,
                                                                             const int x_shape_a,
                                                                             const int h_per_phase,
                                                                             const int padded_len,
                                                                             float *__restrict__ out,
                                                                             const int outW ) {
    _cupy_upfirdn1D<float>( inp, h_trans_flip, up, down, axis, x_shape_a, h_per_phase, padded_len, out, outW );
}

extern "C" __global__ void __launch_bounds__( 512 ) _cupy_upfirdn1D_float64( const double *__restrict__ inp,
                                                                             const double *__restrict__ h_trans_flip,
                                                                             const int up,
                                                                             const int down,
                                                                             const int axis,
                                                                             const int x_shape_a,
                                                                             const int h_per_phase,
                                                                             const int padded_len,
                                                                             double *__restrict__ out,
                                                                             const int outW ) {
    _cupy_upfirdn1D<double>( inp, h_trans_flip, up, down, axis, x_shape_a, h_per_phase, padded_len, out, outW );
}

extern "C" __global__ void __launch_bounds__( 512 )
    _cupy_upfirdn1D_complex64( const thrust::complex<float> *__restrict__ inp,
                               const thrust::complex<float> *__restrict__ h_trans_flip,
                               const int up,
                               const int down,
                               const int axis,
                               const int x_shape_a,
                               const int h_per_phase,
                               const int padded_len,
                               thrust::complex<float> *__restrict__ out,
                               const int outW ) {
    _cupy_upfirdn1D<thrust::complex<float>>(
        inp, h_trans_flip, up, down, axis, x_shape_a, h_per_phase, padded_len, out, outW );
}

extern "C" __global__ void __launch_bounds__( 512 )
    _cupy_upfirdn1D_complex128( const thrust::complex<double> *__restrict__ inp,
                                const thrust::complex<double> *__restrict__ h_trans_flip,
                                const int up,
                                const int down,
                                const int axis,
                                const int x_shape_a,
                                const int h_per_phase,
                                const int padded_len,
                                thrust::complex<double> *__restrict__ out,
                                const int outW ) {
    _cupy_upfirdn1D<thrust::complex<double>>(
        inp, h_trans_flip, up, down, axis, x_shape_a, h_per_phase, padded_len, out, outW );
}

///////////////////////////////////////////////////////////////////////////////
//                              UPFIRDN2D                                    //
///////////////////////////////////////////////////////////////////////////////

template<typename T>
__device__ void _cupy_upfirdn2D( const T *__restrict__ inp,
                                 const int inpH,
                                 const T *__restrict__ h_trans_flip,
                                 const int up,
                                 const int down,
                                 const int axis,
                                 const int x_shape_a,
                                 const int h_per_phase,
                                 const int padded_len,
                                 T *__restrict__ out,
                                 const int outW,
                                 const int outH ) {

    const int ty { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
    const int tx { static_cast<int>( blockIdx.y * blockDim.y + threadIdx.y ) };

    const int stride_y { static_cast<int>( blockDim.x * gridDim.x ) };
    const int stride_x { static_cast<int>( blockDim.y * gridDim.y ) };

    for ( int x = tx; x < outH; x += stride_x ) {
        for ( int y = ty; y < outW; y += stride_y ) {
            int x_idx {};
            int h_idx {};

#if ( __CUDACC_VER_MAJOR__ >= 11 ) && ( __CUDACC_VER_MINOR__ >= 2 )
            __builtin_assume( padded_len > 0 );
            __builtin_assume( up > 0 );
            __builtin_assume( down > 0 );
#endif

            if ( axis == 1 ) {
#if ( __CUDACC_VER_MAJOR__ >= 11 ) && ( __CUDACC_VER_MINOR__ >= 2 )
                __builtin_assume( x > 0 );
#endif
                x_idx = ( static_cast<int>( x * down ) / up ) % padded_len;
                h_idx = ( x * down ) % up * h_per_phase;
            } else {
#if ( __CUDACC_VER_MAJOR__ >= 11 ) && ( __CUDACC_VER_MINOR__ >= 2 )
                __builtin_assume( y > 0 );
#endif
                x_idx = ( static_cast<int>( y * down ) / up ) % padded_len;
                h_idx = ( y * down ) % up * h_per_phase;
            }

            int x_conv_idx { x_idx - h_per_phase + 1 };
            if ( x_conv_idx < 0 ) {
                h_idx -= x_conv_idx;
                x_conv_idx = 0;
            }

            T temp {};

            int stop = ( x_shape_a < ( x_idx + 1 ) ) ? x_shape_a : ( x_idx + 1 );

            for ( int x_c = x_conv_idx; x_c < stop; x_c++ ) {
                if ( axis == 1 ) {
                    temp += inp[y * inpH + x_c] * h_trans_flip[h_idx];
                } else {
                    temp += inp[x_c * inpH + x] * h_trans_flip[h_idx];
                }
                h_idx += 1;
            }
            out[y * outH + x] = temp;
        }
    }
}

extern "C" __global__ void __launch_bounds__( 64 ) _cupy_upfirdn2D_float32( const float *__restrict__ inp,
                                                                            const int inpH,
                                                                            const float *__restrict__ h_trans_flip,
                                                                            const int up,
                                                                            const int down,
                                                                            const int axis,
                                                                            const int x_shape_a,
                                                                            const int h_per_phase,
                                                                            const int padded_len,
                                                                            float *__restrict__ out,
                                                                            const int outW,
                                                                            const int outH ) {
    _cupy_upfirdn2D<float>(
        inp, inpH, h_trans_flip, up, down, axis, x_shape_a, h_per_phase, padded_len, out, outW, outH );
}

extern "C" __global__ void __launch_bounds__( 64 ) _cupy_upfirdn2D_float64( const double *__restrict__ inp,
                                                                            const int inpH,
                                                                            const double *__restrict__ h_trans_flip,
                                                                            const int up,
                                                                            const int down,
                                                                            const int axis,
                                                                            const int x_shape_a,
                                                                            const int h_per_phase,
                                                                            const int padded_len,
                                                                            double *__restrict__ out,
                                                                            const int outW,
                                                                            const int outH ) {
    _cupy_upfirdn2D<double>(
        inp, inpH, h_trans_flip, up, down, axis, x_shape_a, h_per_phase, padded_len, out, outW, outH );
}

extern "C" __global__ void __launch_bounds__( 64 )
    _cupy_upfirdn2D_complex64( const thrust::complex<float> *__restrict__ inp,
                               const int inpH,
                               const thrust::complex<float> *__restrict__ h_trans_flip,
                               const int up,
                               const int down,
                               const int axis,
                               const int x_shape_a,
                               const int h_per_phase,
                               const int padded_len,
                               thrust::complex<float> *__restrict__ out,
                               const int outW,
                               const int outH ) {
    _cupy_upfirdn2D<thrust::complex<float>>(
        inp, inpH, h_trans_flip, up, down, axis, x_shape_a, h_per_phase, padded_len, out, outW, outH );
}

extern "C" __global__ void __launch_bounds__( 64 )
    _cupy_upfirdn2D_complex128( const thrust::complex<double> *__restrict__ inp,
                                const int inpH,
                                const thrust::complex<double> *__restrict__ h_trans_flip,
                                const int up,
                                const int down,
                                const int axis,
                                const int x_shape_a,
                                const int h_per_phase,
                                const int padded_len,
                                thrust::complex<double> *__restrict__ out,
                                const int outW,
                                const int outH ) {
    _cupy_upfirdn2D<thrust::complex<double>>(
        inp, inpH, h_trans_flip, up, down, axis, x_shape_a, h_per_phase, padded_len, out, outW, outH );
}
