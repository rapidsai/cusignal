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

///////////////////////////////////////////////////////////////////////////////
//                            FUNCTION POINTERS                              //
///////////////////////////////////////////////////////////////////////////////

template<typename T>
__device__ __forceinline__ bool less( T a, T b ) {
    return ( a < b );
}

template<typename T>
__device__ __forceinline__ bool greater( T a, T b ) {
    return ( a > b );
}

template<typename T>
__device__ __forceinline__ bool less_equal( T a, T b ) {
    return ( a <= b );
}

template<typename T>
__device__ __forceinline__ bool greater_equal( T a, T b ) {
    return ( a >= b );
}

template<typename T>
__device__ __forceinline__ bool equal( T a, T b ) {
    return ( a == b );
}

template<typename T>
__device__ __forceinline__ bool not_equal( T a, T b ) {
    return ( a != b );
}

using op_func_i                      = bool ( * )( int, int );
__device__ op_func_i const func_i[6] = { less, greater, less_equal, greater_equal, equal, not_equal };

using op_func_l                      = bool ( * )( long int, long int );
__device__ op_func_l const func_l[6] = { less, greater, less_equal, greater_equal, equal, not_equal };

using op_func_f                      = bool ( * )( float, float );
__device__ op_func_f const func_f[6] = { less, greater, less_equal, greater_equal, equal, not_equal };

using op_func_d                      = bool ( * )( double, double );
__device__ op_func_d const func_d[6] = { less, greater, less_equal, greater_equal, equal, not_equal };

template<typename T>
__device__ __forceinline__ void clip_plus( const bool &clip, const int &n, T &plus ) {
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

template<typename T>
__device__ __forceinline__ void clip_minus( const bool &clip, const int &n, T &minus ) {
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

        T    data { inp[tid] };
        bool temp { true };

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
    _cupy_boolrelextrema_1D<int, op_func_i>( n, order, clip, inp, results, func_i[comp] );
}

extern "C" __global__ void __launch_bounds__( 512 ) _cupy_boolrelextrema_1D_int64( const int  n,
                                                                                   const int  order,
                                                                                   const bool clip,
                                                                                   const int  comp,
                                                                                   const long int *__restrict__ inp,
                                                                                   bool *__restrict__ results ) {
    _cupy_boolrelextrema_1D<long int, op_func_l>( n, order, clip, inp, results, func_l[comp] );
}

extern "C" __global__ void __launch_bounds__( 512 ) _cupy_boolrelextrema_1D_float32( const int  n,
                                                                                     const int  order,
                                                                                     const bool clip,
                                                                                     const int  comp,
                                                                                     const float *__restrict__ inp,
                                                                                     bool *__restrict__ results ) {
    _cupy_boolrelextrema_1D<float, op_func_f>( n, order, clip, inp, results, func_f[comp] );
}

extern "C" __global__ void __launch_bounds__( 512 ) _cupy_boolrelextrema_1D_float64( const int  n,
                                                                                     const int  order,
                                                                                     const bool clip,
                                                                                     const int  comp,
                                                                                     const double *__restrict__ inp,
                                                                                     bool *__restrict__ results ) {
    _cupy_boolrelextrema_1D<double, op_func_d>( n, order, clip, inp, results, func_d[comp] );
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

        T    data { inp[tid] };
		bool temp { true };
		
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

extern "C" __global__ void __launch_bounds__( 512 ) _cupy_boolrelextrema_2D_int32( const int  in_x,
                                                                                   const int  in_y,
                                                                                   const int  order,
                                                                                   const bool clip,
                                                                                   const int  comp,
                                                                                   const int  axis,
                                                                                   const int *__restrict__ inp,
                                                                                   bool *__restrict__ results ) {
    _cupy_boolrelextrema_2D<int, op_func_i>( in_x, in_y, order, clip, axis, inp, results, func_i[comp] );
}

extern "C" __global__ void __launch_bounds__( 512 ) _cupy_boolrelextrema_2D_int64( const int  in_x,
                                                                                   const int  in_y,
                                                                                   const int  order,
                                                                                   const bool clip,
                                                                                   const int  comp,
                                                                                   const int  axis,
                                                                                   const long int *__restrict__ inp,
                                                                                   bool *__restrict__ results ) {
    _cupy_boolrelextrema_2D<long int, op_func_l>( in_x, in_y, order, clip, axis, inp, results, func_l[comp] );
}

extern "C" __global__ void __launch_bounds__( 512 ) _cupy_boolrelextrema_2D_float32( const int  in_x,
                                                                                     const int  in_y,
                                                                                     const int  order,
                                                                                     const bool clip,
                                                                                     const int  comp,
                                                                                     const int  axis,
                                                                                     const float *__restrict__ inp,
                                                                                     bool *__restrict__ results ) {
    _cupy_boolrelextrema_2D<float, op_func_f>( in_x, in_y, order, clip, axis, inp, results, func_f[comp] );
}

extern "C" __global__ void __launch_bounds__( 512 ) _cupy_boolrelextrema_2D_float64( const int  in_x,
                                                                                     const int  in_y,
                                                                                     const int  order,
                                                                                     const bool clip,
                                                                                     const int  comp,
                                                                                     const int  axis,
                                                                                     const double *__restrict__ inp,
                                                                                     bool *__restrict__ results ) {
    _cupy_boolrelextrema_2D<double, op_func_d>( in_x, in_y, order, clip, axis, inp, results, func_d[comp] );
}