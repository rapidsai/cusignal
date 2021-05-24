// Copyright (c) 2019-2020, NVIDIA CORPORATION.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
