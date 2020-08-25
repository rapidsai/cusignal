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

// #include <thrust/complex.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuComplex.h>

namespace cg = cooperative_groups;

///////////////////////////////////////////////////////////////////////////////
//                            CHANNELIZER                                    //
///////////////////////////////////////////////////////////////////////////////

// template<typename T>
// __device__ void _cupy_channelizer( const int n_chans,
//                                    const int n_taps,
//                                    const int n_pts,
//                                    const T *__restrict__ x,
//                                    const T *__restrict__ h,
//                                    T *__restrict__ y ) {

//     const auto tx { threadIdx.x };
//     const auto ty { threadIdx.y };

//     if ( tx == 0 && ty == 0 )
//         printf( "Hello\n" );
// }

// extern "C" __global__ void __launch_bounds__( 64 ) _cupy_channelizer_float32( const int n_chans,
//                                                                               const int n_taps,
//                                                                               const int n_pts,
//                                                                               const float *__restrict__ x,
//                                                                               const float *__restrict__ h,
//                                                                               float *__restrict__ y ) {
//     _cupy_channelizer<float>( n_chans, n_taps, n_pts, x, h, y );
// }

// extern "C" __global__ void __launch_bounds__( 64 ) _cupy_channelizer_float64( const int n_chans,
//                                                                               const int n_taps,
//                                                                               const int n_pts,
//                                                                               const double *__restrict__ x,
//                                                                               const double *__restrict__ h,
//                                                                               double *__restrict__ y ) {
//     _cupy_channelizer<double>( n_chans, n_taps, n_pts, x, h, y );
// }

// extern "C" __global__ void __launch_bounds__( 64 ) _cupy_channelizer_complex64( const int n_chans,
//                                                                                 const int n_taps,
//                                                                                 const int n_pts,
//                                                                                 const cuFloatComplex *__restrict__ x,
//                                                                                 const cuFloatComplex *__restrict__ h,
//                                                                                 cuFloatComplex *__restrict__ y ) {
//     _cupy_channelizer<cuFloatComplex>( n_chans, n_taps, n_pts, x, h, y );
// }

// extern "C" __global__ void __launch_bounds__( 64 ) _cupy_channelizer_complex128( const int n_chans,
//                                                                                  const int n_taps,
//                                                                                  const int n_pts,
//                                                                                  const cuDoubleComplex *__restrict__
//                                                                                  x, const cuDoubleComplex
//                                                                                  *__restrict__ h, cuDoubleComplex
//                                                                                  *__restrict__ y ) {
//     _cupy_channelizer<cuDoubleComplex>( n_chans, n_taps, n_pts, x, h, y );
// }

// ///////////////////////////////////////////////////////////////////////////////
// //                          CHANNELIZER 8x8                                  //
// ///////////////////////////////////////////////////////////////////////////////

template<typename T, typename U>
__device__ void _cupy_channelizer_8x8( const int n_chans,
                                         const int n_taps,
                                         const int n_pts,
                                         const U *__restrict__ x,
                                         const U *__restrict__ h,
                                         T *__restrict__ y,
                                         U s_h[8][8],
                                         U s_reg[8][8] ) {

    const auto block   = cg::this_thread_block( );
    const auto tile_32 = cg::tiled_partition<32>( block );
    const auto tile_8  = cg::tiled_partition<8>( tile_32 );

    const auto tx { threadIdx.x };
    const auto ty { threadIdx.y };

    // Initialize shared memory
    // Evaluate type at compile-time
    if ( tx < n_chans && ty < n_taps ) {
        if constexpr ( std::is_same_v<U, cuFloatComplex> ) {
            s_h[tx][ty] = cuConjf( h[ty * n_chans + tx] );
        } else if constexpr ( std::is_same_v<U, cuDoubleComplex> ) {
            s_h[tx][ty] = cuConj( h[ty * n_chans + tx] );
        } else {
            s_h[tx][ty] = h[ty * n_chans + tx];
        }
    } else {
        if constexpr ( std::is_same_v<U, cuFloatComplex> ) {
            s_h[tx][ty] = make_cuFloatComplex( 0, 0 );
        } else if constexpr ( std::is_same_v<U, cuDoubleComplex> ) {
            s_h[tx][ty] = make_cuDoubleComplex( 0, 0 );
        } else {
            s_h[tx][ty] = 0.0;
        }
    }

    // Load data
    if ( blockIdx.x >= n_taps ) {
        if ( tx < n_chans && ty < n_taps ) {
            if constexpr ( std::is_same_v<U, cuFloatComplex> ) {
                s_reg[( n_chans - 1 ) - tx][( n_taps - 1 ) - ty] =
                    cuConjf( x[( ( blockIdx.x - n_taps + 1 ) + ty ) * n_chans + tx] );
            } else if constexpr ( std::is_same_v<U, cuDoubleComplex> ) {
                s_reg[( n_chans - 1 ) - tx][( n_taps - 1 ) - ty] =
                    cuConj( x[( ( blockIdx.x - n_taps + 1 ) + ty ) * n_chans + tx] );
            } else {
                s_reg[( n_chans - 1 ) - tx][( n_taps - 1 ) - ty] =
                    x[( ( blockIdx.x - n_taps + 1 ) + ty ) * n_chans + tx];
            }
        }
    } else {
        if ( tx < n_chans && ty <= blockIdx.x ) {
            if constexpr ( std::is_same_v<U, cuFloatComplex> ) {
                s_reg[( n_chans - 1 ) - tx][blockIdx.x - ty] = cuConjf( x[ty * n_chans + tx] );
            } else if constexpr ( std::is_same_v<U, cuDoubleComplex> ) {
                s_reg[( n_chans - 1 ) - tx][blockIdx.x - ty] = cuConj( x[ty * n_chans + tx] );
            } else {
                s_reg[( n_chans - 1 ) - tx][blockIdx.x - ty] = x[ty * n_chans + tx];
            }
        } else {
            if constexpr ( std::is_same_v<U, cuFloatComplex> ) {
                s_reg[tx][ty] = make_cuFloatComplex( 0, 0 );
            } else if constexpr ( std::is_same_v<U, cuDoubleComplex> ) {
                s_reg[tx][ty] = make_cuDoubleComplex( 0, 0 );
            } else {
                s_reg[tx][ty] = 0.0;
            }
        }
    }

    __syncthreads( );

    U temp {};
    U vv {};

    if ( ty < n_chans ) {
        if constexpr ( std::is_same_v<U, cuFloatComplex> ) {
            temp = cuCmulf( s_h[ty][tx], s_reg[ty][tx] );
            vv.x = cg::reduce( tile_8, temp.x, cg::plus<float>( ) );
            vv.y = cg::reduce( tile_8, temp.y, cg::plus<float>( ) );
        } else if constexpr ( std::is_same_v<U, cuDoubleComplex> ) {
            temp = cuCmul( s_h[ty][tx], s_reg[ty][tx] );
            vv.x = cg::reduce( tile_8, temp.x, cg::plus<double>( ) );
            vv.y = cg::reduce( tile_8, temp.y, cg::plus<double>( ) );
        } else {
            temp = s_h[ty][tx] * s_reg[ty][tx];
            vv   = cg::reduce( tile_8, temp, cg::plus<U>( ) );
        }
    }

    if ( tx == 0 && ty < n_chans ) {
        if constexpr ( std::is_same_v<U, cuFloatComplex> ) {
            y[blockIdx.x * n_chans + ty] = vv;
        } else if constexpr ( std::is_same_v<U, cuDoubleComplex> ) {
            y[blockIdx.x * n_chans + ty] = vv;
        } else if constexpr ( std::is_same_v<U, float> ) {
            y[blockIdx.x * n_chans + ty] = make_cuFloatComplex( vv, 0 );
        } else if constexpr ( std::is_same_v<U, double> ) {
            y[blockIdx.x * n_chans + ty] = make_cuDoubleComplex( vv, 0 );
        }
    }
}

extern "C" __global__ void __launch_bounds__( 64 )
    _cupy_channelizer_8x8_float32_complex64( const int n_chans,
                                               const int n_taps,
                                               const int n_pts,
                                               const float *__restrict__ x,
                                               const float *__restrict__ h,
                                               cuFloatComplex *__restrict__ y ) {

    __shared__ float s_h[8][8];
    __shared__ float s_reg[8][8];

    _cupy_channelizer_8x8<cuFloatComplex, float>( n_chans, n_taps, n_pts, x, h, y, s_h, s_reg );
}

extern "C" __global__ void __launch_bounds__( 64 )
    _cupy_channelizer_8x8__complex64_complex64( const int n_chans,
                                                  const int n_taps,
                                                  const int n_pts,
                                                  const cuFloatComplex *__restrict__ x,
                                                  const cuFloatComplex *__restrict__ h,
                                                  cuFloatComplex *__restrict__ y ) {

    __shared__ cuFloatComplex s_h[8][8];
    __shared__ cuFloatComplex s_reg[8][8];

    _cupy_channelizer_8x8<cuFloatComplex, cuFloatComplex>( n_chans, n_taps, n_pts, x, h, y, s_h, s_reg );
}

extern "C" __global__ void __launch_bounds__( 64 )
    _cupy_channelizer_8x8_float64_complex128( const int n_chans,
                                                const int n_taps,
                                                const int n_pts,
                                                const double *__restrict__ x,
                                                const double *__restrict__ h,
                                                cuDoubleComplex *__restrict__ y ) {

    __shared__ double s_h[8][8];
    __shared__ double s_reg[8][8];

    _cupy_channelizer_8x8<cuDoubleComplex, double>( n_chans, n_taps, n_pts, x, h, y, s_h, s_reg );
}

extern "C" __global__ void __launch_bounds__( 64 )
    _cupy_channelizer_8x8_complex128_complex128( const int n_chans,
                                                   const int n_taps,
                                                   const int n_pts,
                                                   const cuDoubleComplex *__restrict__ x,
                                                   const cuDoubleComplex *__restrict__ h,
                                                   cuDoubleComplex *__restrict__ y ) {

    __shared__ cuDoubleComplex s_h[8][8];
    __shared__ cuDoubleComplex s_reg[8][8];

    _cupy_channelizer_8x8<cuDoubleComplex, cuDoubleComplex>( n_chans, n_taps, n_pts, x, h, y, s_h, s_reg );
}

///////////////////////////////////////////////////////////////////////////////
//                        CHANNELIZER 16x16                                  //
///////////////////////////////////////////////////////////////////////////////

template<typename T, typename U>
__device__ void _cupy_channelizer_16x16( const int n_chans,
                                         const int n_taps,
                                         const int n_pts,
                                         const U *__restrict__ x,
                                         const U *__restrict__ h,
                                         T *__restrict__ y,
                                         U s_h[16][16],
                                         U s_reg[16][16] ) {

    const auto block   = cg::this_thread_block( );
    const auto tile_32 = cg::tiled_partition<32>( block );
    const auto tile_16 = cg::tiled_partition<16>( tile_32 );

    const auto tx { threadIdx.x };
    const auto ty { threadIdx.y };

    // Initialize shared memory
    // Evaluate type at compile-time
    if ( tx < n_chans && ty < n_taps ) {
        if constexpr ( std::is_same_v<U, cuFloatComplex> ) {
            s_h[tx][ty] = cuConjf( h[ty * n_chans + tx] );
        } else if constexpr ( std::is_same_v<U, cuDoubleComplex> ) {
            s_h[tx][ty] = cuConj( h[ty * n_chans + tx] );
        } else {
            s_h[tx][ty] = h[ty * n_chans + tx];
        }
    } else {
        if constexpr ( std::is_same_v<U, cuFloatComplex> ) {
            s_h[tx][ty] = make_cuFloatComplex( 0, 0 );
        } else if constexpr ( std::is_same_v<U, cuDoubleComplex> ) {
            s_h[tx][ty] = make_cuDoubleComplex( 0, 0 );
        } else {
            s_h[tx][ty] = 0.0;
        }
    }

    // Load data
    if ( blockIdx.x >= n_taps ) {
        if ( tx < n_chans && ty < n_taps ) {
            if constexpr ( std::is_same_v<U, cuFloatComplex> ) {
                s_reg[( n_chans - 1 ) - tx][( n_taps - 1 ) - ty] =
                    cuConjf( x[( ( blockIdx.x - n_taps + 1 ) + ty ) * n_chans + tx] );
            } else if constexpr ( std::is_same_v<U, cuDoubleComplex> ) {
                s_reg[( n_chans - 1 ) - tx][( n_taps - 1 ) - ty] =
                    cuConj( x[( ( blockIdx.x - n_taps + 1 ) + ty ) * n_chans + tx] );
            } else {
                s_reg[( n_chans - 1 ) - tx][( n_taps - 1 ) - ty] =
                    x[( ( blockIdx.x - n_taps + 1 ) + ty ) * n_chans + tx];
            }
        }
    } else {
        if ( tx < n_chans && ty <= blockIdx.x ) {
            if constexpr ( std::is_same_v<U, cuFloatComplex> ) {
                s_reg[( n_chans - 1 ) - tx][blockIdx.x - ty] = cuConjf( x[ty * n_chans + tx] );
            } else if constexpr ( std::is_same_v<U, cuDoubleComplex> ) {
                s_reg[( n_chans - 1 ) - tx][blockIdx.x - ty] = cuConj( x[ty * n_chans + tx] );
            } else {
                s_reg[( n_chans - 1 ) - tx][blockIdx.x - ty] = x[ty * n_chans + tx];
            }
        } else {
            if constexpr ( std::is_same_v<U, cuFloatComplex> ) {
                s_reg[tx][ty] = make_cuFloatComplex( 0, 0 );
            } else if constexpr ( std::is_same_v<U, cuDoubleComplex> ) {
                s_reg[tx][ty] = make_cuDoubleComplex( 0, 0 );
            } else {
                s_reg[tx][ty] = 0.0;
            }
        }
    }

    __syncthreads( );

    U temp {};
    U vv {};

    if ( ty < n_chans ) {
        if constexpr ( std::is_same_v<U, cuFloatComplex> ) {
            temp = cuCmulf( s_h[ty][tx], s_reg[ty][tx] );
            vv.x = cg::reduce( tile_16, temp.x, cg::plus<float>( ) );
            vv.y = cg::reduce( tile_16, temp.y, cg::plus<float>( ) );
        } else if constexpr ( std::is_same_v<U, cuDoubleComplex> ) {
            temp = cuCmul( s_h[ty][tx], s_reg[ty][tx] );
            vv.x = cg::reduce( tile_16, temp.x, cg::plus<double>( ) );
            vv.y = cg::reduce( tile_16, temp.y, cg::plus<double>( ) );
        } else {
            temp = s_h[ty][tx] * s_reg[ty][tx];
            vv   = cg::reduce( tile_16, temp, cg::plus<U>( ) );
        }
    }

    if ( tx == 0 && ty < n_chans ) {
        if constexpr ( std::is_same_v<U, cuFloatComplex> ) {
            y[blockIdx.x * n_chans + ty] = vv;
        } else if constexpr ( std::is_same_v<U, cuDoubleComplex> ) {
            y[blockIdx.x * n_chans + ty] = vv;
        } else if constexpr ( std::is_same_v<U, float> ) {
            y[blockIdx.x * n_chans + ty] = make_cuFloatComplex( vv, 0 );
        } else if constexpr ( std::is_same_v<U, double> ) {
            y[blockIdx.x * n_chans + ty] = make_cuDoubleComplex( vv, 0 );
        }
    }
}

extern "C" __global__ void __launch_bounds__( 256 )
    _cupy_channelizer_16x16_float32_complex64( const int n_chans,
                                               const int n_taps,
                                               const int n_pts,
                                               const float *__restrict__ x,
                                               const float *__restrict__ h,
                                               cuFloatComplex *__restrict__ y ) {

    __shared__ float s_h[16][16];
    __shared__ float s_reg[16][16];

    _cupy_channelizer_16x16<cuFloatComplex, float>( n_chans, n_taps, n_pts, x, h, y, s_h, s_reg );
}

extern "C" __global__ void __launch_bounds__( 256 )
    _cupy_channelizer_16x16__complex64_complex64( const int n_chans,
                                                  const int n_taps,
                                                  const int n_pts,
                                                  const cuFloatComplex *__restrict__ x,
                                                  const cuFloatComplex *__restrict__ h,
                                                  cuFloatComplex *__restrict__ y ) {

    __shared__ cuFloatComplex s_h[16][16];
    __shared__ cuFloatComplex s_reg[16][16];

    _cupy_channelizer_16x16<cuFloatComplex, cuFloatComplex>( n_chans, n_taps, n_pts, x, h, y, s_h, s_reg );
}

extern "C" __global__ void __launch_bounds__( 256 )
    _cupy_channelizer_16x16_float64_complex128( const int n_chans,
                                                const int n_taps,
                                                const int n_pts,
                                                const double *__restrict__ x,
                                                const double *__restrict__ h,
                                                cuDoubleComplex *__restrict__ y ) {

    __shared__ double s_h[16][16];
    __shared__ double s_reg[16][16];

    _cupy_channelizer_16x16<cuDoubleComplex, double>( n_chans, n_taps, n_pts, x, h, y, s_h, s_reg );
}

extern "C" __global__ void __launch_bounds__( 256 )
    _cupy_channelizer_16x16_complex128_complex128( const int n_chans,
                                                   const int n_taps,
                                                   const int n_pts,
                                                   const cuDoubleComplex *__restrict__ x,
                                                   const cuDoubleComplex *__restrict__ h,
                                                   cuDoubleComplex *__restrict__ y ) {

    __shared__ cuDoubleComplex s_h[16][16];
    __shared__ cuDoubleComplex s_reg[16][16];

    _cupy_channelizer_16x16<cuDoubleComplex, cuDoubleComplex>( n_chans, n_taps, n_pts, x, h, y, s_h, s_reg );
}

///////////////////////////////////////////////////////////////////////////////
//                        CHANNELIZER 32x32                                  //
///////////////////////////////////////////////////////////////////////////////

template<typename T, typename U>
__device__ void _cupy_channelizer_32x32( const int n_chans,
                                         const int n_taps,
                                         const int n_pts,
                                         const U *__restrict__ x,
                                         const U *__restrict__ h,
                                         T *__restrict__ y,
                                         U s_h[32][32],
                                         U s_reg[32][32] ) {

    const auto block   = cg::this_thread_block( );
    const auto tile_32 = cg::tiled_partition<32>( block );

    const auto tx { threadIdx.x };
    const auto ty { threadIdx.y };

    // Initialize shared memory
    // Evaluate type at compile-time
    if ( tx < n_chans && ty < n_taps ) {
        if constexpr ( std::is_same_v<U, cuFloatComplex> ) {
            s_h[tx][ty] = cuConjf( h[ty * n_chans + tx] );
        } else if constexpr ( std::is_same_v<U, cuDoubleComplex> ) {
            s_h[tx][ty] = cuConj( h[ty * n_chans + tx] );
        } else {
            s_h[tx][ty] = h[ty * n_chans + tx];
        }
    } else {
        if constexpr ( std::is_same_v<U, cuFloatComplex> ) {
            s_h[tx][ty] = make_cuFloatComplex( 0, 0 );
        } else if constexpr ( std::is_same_v<U, cuDoubleComplex> ) {
            s_h[tx][ty] = make_cuDoubleComplex( 0, 0 );
        } else {
            s_h[tx][ty] = 0.0;
        }
    }

    // Load data
    if ( blockIdx.x >= n_taps ) {
        if ( tx < n_chans && ty < n_taps ) {
            if constexpr ( std::is_same_v<U, cuFloatComplex> ) {
                s_reg[( n_chans - 1 ) - tx][( n_taps - 1 ) - ty] =
                    cuConjf( x[( ( blockIdx.x - n_taps + 1 ) + ty ) * n_chans + tx] );
            } else if constexpr ( std::is_same_v<U, cuDoubleComplex> ) {
                s_reg[( n_chans - 1 ) - tx][( n_taps - 1 ) - ty] =
                    cuConj( x[( ( blockIdx.x - n_taps + 1 ) + ty ) * n_chans + tx] );
            } else {
                s_reg[( n_chans - 1 ) - tx][( n_taps - 1 ) - ty] =
                    x[( ( blockIdx.x - n_taps + 1 ) + ty ) * n_chans + tx];
            }
        }
    } else {
        if ( tx < n_chans && ty <= blockIdx.x ) {
            if constexpr ( std::is_same_v<U, cuFloatComplex> ) {
                s_reg[( n_chans - 1 ) - tx][blockIdx.x - ty] = cuConjf( x[ty * n_chans + tx] );
            } else if constexpr ( std::is_same_v<U, cuDoubleComplex> ) {
                s_reg[( n_chans - 1 ) - tx][blockIdx.x - ty] = cuConj( x[ty * n_chans + tx] );
            } else {
                s_reg[( n_chans - 1 ) - tx][blockIdx.x - ty] = x[ty * n_chans + tx];
            }
        } else {
            if constexpr ( std::is_same_v<U, cuFloatComplex> ) {
                s_reg[tx][ty] = make_cuFloatComplex( 0, 0 );
            } else if constexpr ( std::is_same_v<U, cuDoubleComplex> ) {
                s_reg[tx][ty] = make_cuDoubleComplex( 0, 0 );
            } else {
                s_reg[tx][ty] = 0.0;
            }
        }
    }

    __syncthreads( );

    U temp {};
    U vv {};

    if ( ty < n_chans ) {
        if constexpr ( std::is_same_v<U, cuFloatComplex> ) {
            temp = cuCmulf( s_h[ty][tx], s_reg[ty][tx] );
            vv.x = cg::reduce( tile_32, temp.x, cg::plus<float>( ) );
            vv.y = cg::reduce( tile_32, temp.y, cg::plus<float>( ) );
        } else if constexpr ( std::is_same_v<U, cuDoubleComplex> ) {
            temp = cuCmul( s_h[ty][tx], s_reg[ty][tx] );
            vv.x = cg::reduce( tile_32, temp.x, cg::plus<double>( ) );
            vv.y = cg::reduce( tile_32, temp.y, cg::plus<double>( ) );
        } else {
            temp = s_h[ty][tx] * s_reg[ty][tx];
            vv   = cg::reduce( tile_32, temp, cg::plus<U>( ) );
        }
    }

    if ( tx == 0 && ty < n_chans ) {
        if constexpr ( std::is_same_v<U, cuFloatComplex> ) {
            y[blockIdx.x * n_chans + ty] = vv;
        } else if constexpr ( std::is_same_v<U, cuDoubleComplex> ) {
            y[blockIdx.x * n_chans + ty] = vv;
        } else if constexpr ( std::is_same_v<U, float> ) {
            y[blockIdx.x * n_chans + ty] = make_cuFloatComplex( vv, 0 );
        } else if constexpr ( std::is_same_v<U, double> ) {
            y[blockIdx.x * n_chans + ty] = make_cuDoubleComplex( vv, 0 );
        }
    }
}

extern "C" __global__ void __launch_bounds__( 1024 )
    _cupy_channelizer_32x32_float32_complex64( const int n_chans,
                                               const int n_taps,
                                               const int n_pts,
                                               const float *__restrict__ x,
                                               const float *__restrict__ h,
                                               cuFloatComplex *__restrict__ y ) {

    __shared__ float s_h[32][32];
    __shared__ float s_reg[32][32];

    _cupy_channelizer_32x32<cuFloatComplex, float>( n_chans, n_taps, n_pts, x, h, y, s_h, s_reg );
}

extern "C" __global__ void __launch_bounds__( 1024 )
    _cupy_channelizer_32x32_complex64_complex64( const int n_chans,
                                                 const int n_taps,
                                                 const int n_pts,
                                                 const cuFloatComplex *__restrict__ x,
                                                 const cuFloatComplex *__restrict__ h,
                                                 cuFloatComplex *__restrict__ y ) {

    __shared__ cuFloatComplex s_h[32][32];
    __shared__ cuFloatComplex s_reg[32][32];

    _cupy_channelizer_32x32<cuFloatComplex, cuFloatComplex>( n_chans, n_taps, n_pts, x, h, y, s_h, s_reg );
}

extern "C" __global__ void __launch_bounds__( 1024 )
    _cupy_channelizer_32x32_float64_complex128( const int n_chans,
                                                const int n_taps,
                                                const int n_pts,
                                                const double *__restrict__ x,
                                                const double *__restrict__ h,
                                                cuDoubleComplex *__restrict__ y ) {

    __shared__ double s_h[32][32];
    __shared__ double s_reg[32][32];

    _cupy_channelizer_32x32<cuDoubleComplex, double>( n_chans, n_taps, n_pts, x, h, y, s_h, s_reg );
}

extern "C" __global__ void __launch_bounds__( 1024 )
    _cupy_channelizer_32x32_complex128_complex128( const int n_chans,
                                                   const int n_taps,
                                                   const int n_pts,
                                                   const cuDoubleComplex *__restrict__ x,
                                                   const cuDoubleComplex *__restrict__ h,
                                                   cuDoubleComplex *__restrict__ y ) {

    __shared__ cuDoubleComplex s_h[32][32];
    __shared__ cuDoubleComplex s_reg[32][32];

    _cupy_channelizer_32x32<cuDoubleComplex, cuDoubleComplex>( n_chans, n_taps, n_pts, x, h, y, s_h, s_reg );
}