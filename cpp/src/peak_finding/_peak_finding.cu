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
#include <nvfunctional>

///////////////////////////////////////////////////////////////////////////////
//                            FUNCTION POINTERS                              //
///////////////////////////////////////////////////////////////////////////////

template<typename T>
__device__ __forceinline__ bool less( const T &a, const T &b ) {
    return ( a < b );
}

template<typename T>
__device__ __forceinline__ bool greater( const T &a, const T &b ) {
    return ( a > b );
}

template<typename T>
__device__ __forceinline__ bool less_equal( const T &a, const T &b ) {
    return ( a <= b );
}

template<typename T>
__device__ __forceinline__ bool greater_equal( const T &a, const T &b ) {
    return ( a >= b );
}

template<typename T>
__device__ __forceinline__ bool equal( const T &a, const T &b ) {
    return ( a == b );
}

template<typename T>
__device__ __forceinline__ bool not_equal( const T &a, const T &b ) {
    return ( a != b );
}

template<typename T>
using op_func = bool ( * )( const T &, const T & );

__device__ op_func<int> const func_i[6]      = { less, greater, less_equal, greater_equal, equal, not_equal };
__device__ op_func<long int> const func_l[6] = { less, greater, less_equal, greater_equal, equal, not_equal };
__device__ op_func<float> const func_f[6]    = { less, greater, less_equal, greater_equal, equal, not_equal };
__device__ op_func<double> const func_d[6]   = { less, greater, less_equal, greater_equal, equal, not_equal };

///////////////////////////////////////////////////////////////////////////////
//                              HELPER FUNCTIONS                             //
///////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ void clip_plus( const bool &clip, const int &n, int &plus ) {
    if ( clip ) {
        if ( plus >= n ) {
            plus = n - 1;
        }
    } else {
        if ( plus >= n ) {
            plus -= n;
        }
    }
}

__device__ __forceinline__ void clip_minus( const bool &clip, const int &n, int &minus ) {
    if ( clip ) {
        if ( minus < 0 ) {
            minus = 0;
        }
    } else {
        if ( minus < 0 ) {
            minus += n;
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
//                          BOOLRELEXTREMA 1D                                //
///////////////////////////////////////////////////////////////////////////////

template<typename T, class U>
__device__ void _cupy_boolrelextrema_1D( const int  n,
                                         const int  order,
                                         const bool clip,
                                         const T *__restrict__ inp,
                                         bool *__restrict__ results,
                                         U func ) {

    const int tx { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
    const int stride { static_cast<int>( blockDim.x * gridDim.x ) };

    for ( int tid = tx; tid < n; tid += stride ) {

        const T data { inp[tid] };
        bool    temp { true };

        for ( int o = 1; o < ( order + 1 ); o++ ) {
            int plus { tid + o };
            int minus { tid - o };

            clip_plus( clip, n, plus );
            clip_minus( clip, n, minus );

            temp &= func( data, inp[plus] );
            temp &= func( data, inp[minus] );
        }
        results[tid] = temp;
    }
}

extern "C" __global__ void __launch_bounds__( 512 ) _cupy_boolrelextrema_1D_int32( const int  n,
                                                                                   const int  order,
                                                                                   const bool clip,
                                                                                   const int  comp,
                                                                                   const int *__restrict__ inp,
                                                                                   bool *__restrict__ results ) {
    _cupy_boolrelextrema_1D<int, op_func<int>>( n, order, clip, inp, results, func_i[comp] );
}

extern "C" __global__ void __launch_bounds__( 512 ) _cupy_boolrelextrema_1D_int64( const int  n,
                                                                                   const int  order,
                                                                                   const bool clip,
                                                                                   const int  comp,
                                                                                   const long int *__restrict__ inp,
                                                                                   bool *__restrict__ results ) {
    _cupy_boolrelextrema_1D<long int, op_func<long int>>( n, order, clip, inp, results, func_l[comp] );
}

extern "C" __global__ void __launch_bounds__( 512 ) _cupy_boolrelextrema_1D_float32( const int  n,
                                                                                     const int  order,
                                                                                     const bool clip,
                                                                                     const int  comp,
                                                                                     const float *__restrict__ inp,
                                                                                     bool *__restrict__ results ) {
    _cupy_boolrelextrema_1D<float, op_func<float>>( n, order, clip, inp, results, func_f[comp] );
}

extern "C" __global__ void __launch_bounds__( 512 ) _cupy_boolrelextrema_1D_float64( const int  n,
                                                                                     const int  order,
                                                                                     const bool clip,
                                                                                     const int  comp,
                                                                                     const double *__restrict__ inp,
                                                                                     bool *__restrict__ results ) {
    _cupy_boolrelextrema_1D<double, op_func<double>>( n, order, clip, inp, results, func_d[comp] );
}

///////////////////////////////////////////////////////////////////////////////
//                          BOOLRELEXTREMA 2D                                //
///////////////////////////////////////////////////////////////////////////////

template<typename T, class U>
__device__ void _cupy_boolrelextrema_2D( const int  in_x,
                                         const int  in_y,
                                         const int  order,
                                         const bool clip,
                                         const int  axis,
                                         const T *__restrict__ inp,
                                         bool *__restrict__ results,
                                         U func ) {

    const int ty { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
    const int tx { static_cast<int>( blockIdx.y * blockDim.y + threadIdx.y ) };

    if ( ( tx < in_y ) && ( ty < in_x ) ) {
        int tid { tx * in_x + ty };

        const T data { inp[tid] };
        bool    temp { true };

        for ( int o = 1; o < ( order + 1 ); o++ ) {

            int plus {};
            int minus {};

            if ( axis == 0 ) {
                plus  = tx + o;
                minus = tx - o;

                clip_plus( clip, in_y, plus );
                clip_minus( clip, in_y, minus );

                plus  = plus * in_x + ty;
                minus = minus * in_x + ty;
            } else {
                plus  = ty + o;
                minus = ty - o;

                clip_plus( clip, in_x, plus );
                clip_minus( clip, in_x, minus );

                plus  = tx * in_x + plus;
                minus = tx * in_x + minus;
            }

            temp &= func( data, inp[plus] );
            temp &= func( data, inp[minus] );
        }
        results[tid] = temp;
    }
}

extern "C" __global__ void __launch_bounds__( 256 ) _cupy_boolrelextrema_2D_int32( const int  in_x,
                                                                                   const int  in_y,
                                                                                   const int  order,
                                                                                   const bool clip,
                                                                                   const int  comp,
                                                                                   const int  axis,
                                                                                   const int *__restrict__ inp,
                                                                                   bool *__restrict__ results ) {
    _cupy_boolrelextrema_2D<int, op_func<int>>( in_x, in_y, order, clip, axis, inp, results, func_i[comp] );
}

extern "C" __global__ void __launch_bounds__( 256 ) _cupy_boolrelextrema_2D_int64( const int  in_x,
                                                                                   const int  in_y,
                                                                                   const int  order,
                                                                                   const bool clip,
                                                                                   const int  comp,
                                                                                   const int  axis,
                                                                                   const long int *__restrict__ inp,
                                                                                   bool *__restrict__ results ) {
    _cupy_boolrelextrema_2D<long int, op_func<long int>>( in_x, in_y, order, clip, axis, inp, results, func_l[comp] );
}

extern "C" __global__ void __launch_bounds__( 256 ) _cupy_boolrelextrema_2D_float32( const int  in_x,
                                                                                     const int  in_y,
                                                                                     const int  order,
                                                                                     const bool clip,
                                                                                     const int  comp,
                                                                                     const int  axis,
                                                                                     const float *__restrict__ inp,
                                                                                     bool *__restrict__ results ) {
    _cupy_boolrelextrema_2D<float, op_func<float>>( in_x, in_y, order, clip, axis, inp, results, func_f[comp] );
}

extern "C" __global__ void __launch_bounds__( 256 ) _cupy_boolrelextrema_2D_float64( const int  in_x,
                                                                                     const int  in_y,
                                                                                     const int  order,
                                                                                     const bool clip,
                                                                                     const int  comp,
                                                                                     const int  axis,
                                                                                     const double *__restrict__ inp,
                                                                                     bool *__restrict__ results ) {
    _cupy_boolrelextrema_2D<double, op_func<double>>( in_x, in_y, order, clip, axis, inp, results, func_d[comp] );
}