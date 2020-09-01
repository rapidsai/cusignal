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

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <thrust/complex.h>

namespace cg = cooperative_groups;

///////////////////////////////////////////////////////////////////////////////
//                          CHANNELIZER 8x8                                  //
///////////////////////////////////////////////////////////////////////////////

// T is input type
// U is output type
template<typename T, typename U, int M = 8, int WARPSIZE = 32>
__device__ void _cupy_channelizer_8x8( const int n_chans,
                                       const int n_taps,
                                       const int n_pts,
                                       const T *__restrict__ x,
                                       const T *__restrict__ h,
                                       U *__restrict__ y,
                                       T s_h[M][M],
                                       T s_reg[M][M] ) {

    const auto block { cg::this_thread_block( ) };
    const auto tile_32 { cg::tiled_partition<WARPSIZE>( block ) };
    const auto tile { cg::tiled_partition<M>( tile_32 ) };

    const auto btx { blockIdx.x * blockDim.x + threadIdx.x };

    const auto tx { threadIdx.x };
    const auto ty { threadIdx.y };

    // Initialize shared memory
    // Evaluate type at compile-time
    if ( btx < n_chans && ty < n_taps ) {
        if constexpr ( std::is_same_v<T, thrust::complex<float>> || std::is_same_v<T, thrust::complex<double>> ) {
            s_h[tx][ty] = thrust::conj( h[ty * n_chans + btx] );
        } else {
            s_h[tx][ty] = h[ty * n_chans + btx];
        }
    } else {
        if constexpr ( std::is_same_v<T, thrust::complex<float>> || std::is_same_v<T, thrust::complex<double>> ) {
            s_h[tx][ty] = T( 0, 0 );
        } else {
            s_h[tx][ty] = 0.0;
        }
    }

    for ( auto bid = blockIdx.y; bid < n_pts; bid += blockDim.y ) {
        // Load data
        if ( bid >= n_taps ) {
            if ( btx < n_chans && ty < n_taps ) {
                if constexpr ( std::is_same_v<T, thrust::complex<float>> ||
                               std::is_same_v<T, thrust::complex<double>> ) {
                    s_reg[tx][( n_taps - 1 ) - ty] =
                        thrust::conj( x[( ( bid - n_taps + 1 ) + ty ) * n_chans + ( n_chans - 1 - btx )] );
                } else {
                    s_reg[tx][( n_taps - 1 ) - ty] = x[( ( bid - n_taps + 1 ) + ty ) * n_chans + ( n_chans - 1 - btx )];
                }
            }
        } else {
            if ( btx < n_chans && ty <= bid ) {
                if constexpr ( std::is_same_v<T, thrust::complex<float>> ||
                               std::is_same_v<T, thrust::complex<double>> ) {
                    s_reg[tx][bid - ty] = thrust::conj( x[ty * n_chans + ( n_chans - 1 - btx )] );
                } else {
                    s_reg[tx][bid - ty] = x[ty * n_chans + ( n_chans - 1 - btx )];
                }
            } else {
                if constexpr ( std::is_same_v<T, thrust::complex<float>> ||
                               std::is_same_v<T, thrust::complex<double>> ) {
                    s_reg[tx][ty] = T( 0, 0 );
                } else {
                    s_reg[tx][ty] = 0.0;
                }
            }
        }

        __syncthreads( );

        U temp {};
        U vv {};

        // Perform compute
        if ( ( blockIdx.x * M + ty ) < n_chans ) {
            temp = s_h[ty][tx] * s_reg[ty][tx];
            vv.real( cg::reduce( tile, temp.real( ), cg::plus<typename U::value_type>( ) ) );
            vv.imag( cg::reduce( tile, temp.imag( ), cg::plus<typename U::value_type>( ) ) );
        }

        // Store output
        if ( tx == 0 && ( blockIdx.x * M + ty ) < n_chans ) {
            y[bid * n_chans + ( blockIdx.x * M + ty )] = vv;
        }
    }
}

extern "C" __global__ void __launch_bounds__( 64 )
    _cupy_channelizer_8x8_float32_complex64( const int n_chans,
                                             const int n_taps,
                                             const int n_pts,
                                             const float *__restrict__ x,
                                             const float *__restrict__ h,
                                             thrust::complex<float> *__restrict__ y ) {

    __shared__ float s_h[8][8];
    __shared__ float s_reg[8][8];

    _cupy_channelizer_8x8<float, thrust::complex<float>>( n_chans, n_taps, n_pts, x, h, y, s_h, s_reg );
}

extern "C" __global__ void __launch_bounds__( 64 )
    _cupy_channelizer_8x8_complex64_complex64( const int n_chans,
                                               const int n_taps,
                                               const int n_pts,
                                               const thrust::complex<float> *__restrict__ x,
                                               const thrust::complex<float> *__restrict__ h,
                                               thrust::complex<float> *__restrict__ y ) {

    __shared__ thrust::complex<float> s_h[8][8];
    __shared__ thrust::complex<float> s_reg[8][8];

    _cupy_channelizer_8x8<thrust::complex<float>, thrust::complex<float>>(
        n_chans, n_taps, n_pts, x, h, y, s_h, s_reg );
}

extern "C" __global__ void __launch_bounds__( 64 )
    _cupy_channelizer_8x8_float64_complex128( const int n_chans,
                                              const int n_taps,
                                              const int n_pts,
                                              const double *__restrict__ x,
                                              const double *__restrict__ h,
                                              thrust::complex<double> *__restrict__ y ) {

    __shared__ double s_h[8][8];
    __shared__ double s_reg[8][8];

    _cupy_channelizer_8x8<double, thrust::complex<double>>( n_chans, n_taps, n_pts, x, h, y, s_h, s_reg );
}

extern "C" __global__ void __launch_bounds__( 64 )
    _cupy_channelizer_8x8_complex128_complex128( const int n_chans,
                                                 const int n_taps,
                                                 const int n_pts,
                                                 const thrust::complex<double> *__restrict__ x,
                                                 const thrust::complex<double> *__restrict__ h,
                                                 thrust::complex<double> *__restrict__ y ) {

    __shared__ thrust::complex<double> s_h[8][8];
    __shared__ thrust::complex<double> s_reg[8][8];

    _cupy_channelizer_8x8<thrust::complex<double>, thrust::complex<double>>(
        n_chans, n_taps, n_pts, x, h, y, s_h, s_reg );
}

///////////////////////////////////////////////////////////////////////////////
//                        CHANNELIZER 16x16                                  //
///////////////////////////////////////////////////////////////////////////////

// T is input type
// U is output type
template<typename T, typename U, int M = 16, int WARPSIZE = 32>
__device__ void _cupy_channelizer_16x16( const int n_chans,
                                         const int n_taps,
                                         const int n_pts,
                                         const T *__restrict__ x,
                                         const T *__restrict__ h,
                                         U *__restrict__ y,
                                         T s_h[M][M],
                                         T s_reg[M][M] ) {

    const auto block { cg::this_thread_block( ) };
    const auto tile_32 { cg::tiled_partition<WARPSIZE>( block ) };
    const auto tile { cg::tiled_partition<M>( tile_32 ) };

    const auto btx { blockIdx.x * blockDim.x + threadIdx.x };

    const auto tx { threadIdx.x };
    const auto ty { threadIdx.y };

    // Initialize shared memory
    // Evaluate type at compile-time
    if ( btx < n_chans && ty < n_taps ) {
        if constexpr ( std::is_same_v<T, thrust::complex<float>> || std::is_same_v<T, thrust::complex<double>> ) {
            s_h[tx][ty] = thrust::conj( h[ty * n_chans + btx] );
        } else {
            s_h[tx][ty] = h[ty * n_chans + btx];
        }
    } else {
        if constexpr ( std::is_same_v<T, thrust::complex<float>> || std::is_same_v<T, thrust::complex<double>> ) {
            s_h[tx][ty] = T( 0, 0 );
        } else {
            s_h[tx][ty] = 0.0;
        }
    }

    for ( auto bid = blockIdx.y; bid < n_pts; bid += blockDim.y ) {
        // Load data
        if ( bid >= n_taps ) {
            if ( btx < n_chans && ty < n_taps ) {
                if constexpr ( std::is_same_v<T, thrust::complex<float>> ||
                               std::is_same_v<T, thrust::complex<double>> ) {
                    s_reg[tx][( n_taps - 1 ) - ty] =
                        thrust::conj( x[( ( bid - n_taps + 1 ) + ty ) * n_chans + ( n_chans - 1 - btx )] );
                } else {
                    s_reg[tx][( n_taps - 1 ) - ty] = x[( ( bid - n_taps + 1 ) + ty ) * n_chans + ( n_chans - 1 - btx )];
                }
            }
        } else {
            if ( btx < n_chans && ty <= bid ) {
                if constexpr ( std::is_same_v<T, thrust::complex<float>> ||
                               std::is_same_v<T, thrust::complex<double>> ) {
                    s_reg[tx][bid - ty] = thrust::conj( x[ty * n_chans + ( n_chans - 1 - btx )] );
                } else {
                    s_reg[tx][bid - ty] = x[ty * n_chans + ( n_chans - 1 - btx )];
                }
            } else {
                if constexpr ( std::is_same_v<T, thrust::complex<float>> ||
                               std::is_same_v<T, thrust::complex<double>> ) {
                    s_reg[tx][ty] = T( 0, 0 );
                } else {
                    s_reg[tx][ty] = 0.0;
                }
            }
        }

        __syncthreads( );

        U temp {};
        U vv {};

        // Perform compute
        if ( ( blockIdx.x * M + ty ) < n_chans ) {
            temp = s_h[ty][tx] * s_reg[ty][tx];
            vv.real( cg::reduce( tile, temp.real( ), cg::plus<typename U::value_type>( ) ) );
            vv.imag( cg::reduce( tile, temp.imag( ), cg::plus<typename U::value_type>( ) ) );
        }

        // Store output
        if ( tx == 0 && ( blockIdx.x * M + ty ) < n_chans ) {
            y[bid * n_chans + ( blockIdx.x * M + ty )] = vv;
        }
    }
}

extern "C" __global__ void __launch_bounds__( 256 )
    _cupy_channelizer_16x16_float32_complex64( const int n_chans,
                                               const int n_taps,
                                               const int n_pts,
                                               const float *__restrict__ x,
                                               const float *__restrict__ h,
                                               thrust::complex<float> *__restrict__ y ) {

    __shared__ float s_h[16][16];
    __shared__ float s_reg[16][16];

    _cupy_channelizer_16x16<float, thrust::complex<float>>( n_chans, n_taps, n_pts, x, h, y, s_h, s_reg );
}

extern "C" __global__ void __launch_bounds__( 256 )
    _cupy_channelizer_16x16_complex64_complex64( const int n_chans,
                                                 const int n_taps,
                                                 const int n_pts,
                                                 const thrust::complex<float> *__restrict__ x,
                                                 const thrust::complex<float> *__restrict__ h,
                                                 thrust::complex<float> *__restrict__ y ) {

    __shared__ thrust::complex<float> s_h[16][16];
    __shared__ thrust::complex<float> s_reg[16][16];

    _cupy_channelizer_16x16<thrust::complex<float>, thrust::complex<float>>(
        n_chans, n_taps, n_pts, x, h, y, s_h, s_reg );
}

extern "C" __global__ void __launch_bounds__( 256 )
    _cupy_channelizer_16x16_float64_complex128( const int n_chans,
                                                const int n_taps,
                                                const int n_pts,
                                                const double *__restrict__ x,
                                                const double *__restrict__ h,
                                                thrust::complex<double> *__restrict__ y ) {

    __shared__ double s_h[16][16];
    __shared__ double s_reg[16][16];

    _cupy_channelizer_16x16<double, thrust::complex<double>>( n_chans, n_taps, n_pts, x, h, y, s_h, s_reg );
}

extern "C" __global__ void __launch_bounds__( 256 )
    _cupy_channelizer_16x16_complex128_complex128( const int n_chans,
                                                   const int n_taps,
                                                   const int n_pts,
                                                   const thrust::complex<double> *__restrict__ x,
                                                   const thrust::complex<double> *__restrict__ h,
                                                   thrust::complex<double> *__restrict__ y ) {

    __shared__ thrust::complex<double> s_h[16][16];
    __shared__ thrust::complex<double> s_reg[16][16];

    _cupy_channelizer_16x16<thrust::complex<double>, thrust::complex<double>>(
        n_chans, n_taps, n_pts, x, h, y, s_h, s_reg );
}

///////////////////////////////////////////////////////////////////////////////
//                        CHANNELIZER 32x32                                  //
///////////////////////////////////////////////////////////////////////////////

// T is input type
// U is output type
template<typename T, typename U, int M = 32, int WARPSIZE = 32>
__device__ void _cupy_channelizer_32x32( const int n_chans,
                                         const int n_taps,
                                         const int n_pts,
                                         const T *__restrict__ x,
                                         const T *__restrict__ h,
                                         U *__restrict__ y,
                                         T s_h[M][M],
                                         T s_reg[M][M] ) {

    const auto block { cg::this_thread_block( ) };
    const auto tile { cg::tiled_partition<WARPSIZE>( block ) };

    const auto btx { blockIdx.x * blockDim.x + threadIdx.x };

    const auto tx { threadIdx.x };
    const auto ty { threadIdx.y };

    // Initialize shared memory
    // Evaluate type at compile-time
    if ( btx < n_chans && ty < n_taps ) {
        if constexpr ( std::is_same_v<T, thrust::complex<float>> || std::is_same_v<T, thrust::complex<double>> ) {
            s_h[tx][ty] = thrust::conj( h[ty * n_chans + btx] );
        } else {
            s_h[tx][ty] = h[ty * n_chans + btx];
        }
    } else {
        if constexpr ( std::is_same_v<T, thrust::complex<float>> || std::is_same_v<T, thrust::complex<double>> ) {
            s_h[tx][ty] = T( 0, 0 );
        } else {
            s_h[tx][ty] = 0.0;
        }
    }

    for ( auto bid = blockIdx.y; bid < n_pts; bid += blockDim.y ) {
        // Load data
        if ( bid >= n_taps ) {
            if ( btx < n_chans && ty < n_taps ) {
                if constexpr ( std::is_same_v<T, thrust::complex<float>> ||
                               std::is_same_v<T, thrust::complex<double>> ) {
                    s_reg[tx][( n_taps - 1 ) - ty] =
                        thrust::conj( x[( ( bid - n_taps + 1 ) + ty ) * n_chans + ( n_chans - 1 - btx )] );
                } else {
                    s_reg[tx][( n_taps - 1 ) - ty] = x[( ( bid - n_taps + 1 ) + ty ) * n_chans + ( n_chans - 1 - btx )];
                }
            }
        } else {
            if ( btx < n_chans && ty <= bid ) {
                if constexpr ( std::is_same_v<T, thrust::complex<float>> ||
                               std::is_same_v<T, thrust::complex<double>> ) {
                    s_reg[tx][bid - ty] = thrust::conj( x[ty * n_chans + ( n_chans - 1 - btx )] );
                } else {
                    s_reg[tx][bid - ty] = x[ty * n_chans + ( n_chans - 1 - btx )];
                }
            } else {
                if constexpr ( std::is_same_v<T, thrust::complex<float>> ||
                               std::is_same_v<T, thrust::complex<double>> ) {
                    s_reg[tx][ty] = T( 0, 0 );
                } else {
                    s_reg[tx][ty] = 0.0;
                }
            }
        }

        __syncthreads( );

        U temp {};
        U vv {};

        // Perform compute
        if ( ( blockIdx.x * M + ty ) < n_chans ) {
            temp = s_h[ty][tx] * s_reg[ty][tx];
            vv.real( cg::reduce( tile, temp.real( ), cg::plus<typename U::value_type>( ) ) );
            vv.imag( cg::reduce( tile, temp.imag( ), cg::plus<typename U::value_type>( ) ) );
        }

        // Store output
        if ( tx == 0 && ( blockIdx.x * M + ty ) < n_chans ) {
            y[bid * n_chans + ( blockIdx.x * M + ty )] = vv;
        }
    }
}

extern "C" __global__ void __launch_bounds__( 1024 )
    _cupy_channelizer_32x32_float32_complex64( const int n_chans,
                                               const int n_taps,
                                               const int n_pts,
                                               const float *__restrict__ x,
                                               const float *__restrict__ h,
                                               thrust::complex<float> *__restrict__ y ) {

    __shared__ float s_h[32][32];
    __shared__ float s_reg[32][32];

    _cupy_channelizer_32x32<float, thrust::complex<float>>( n_chans, n_taps, n_pts, x, h, y, s_h, s_reg );
}

extern "C" __global__ void __launch_bounds__( 1024 )
    _cupy_channelizer_32x32_complex64_complex64( const int n_chans,
                                                 const int n_taps,
                                                 const int n_pts,
                                                 const thrust::complex<float> *__restrict__ x,
                                                 const thrust::complex<float> *__restrict__ h,
                                                 thrust::complex<float> *__restrict__ y ) {

    __shared__ thrust::complex<float> s_h[32][32];
    __shared__ thrust::complex<float> s_reg[32][32];

    _cupy_channelizer_32x32<thrust::complex<float>, thrust::complex<float>>(
        n_chans, n_taps, n_pts, x, h, y, s_h, s_reg );
}

extern "C" __global__ void __launch_bounds__( 1024 )
    _cupy_channelizer_32x32_float64_complex128( const int n_chans,
                                                const int n_taps,
                                                const int n_pts,
                                                const double *__restrict__ x,
                                                const double *__restrict__ h,
                                                thrust::complex<double> *__restrict__ y ) {

    __shared__ double s_h[32][32];
    __shared__ double s_reg[32][32];

    _cupy_channelizer_32x32<double, thrust::complex<double>>( n_chans, n_taps, n_pts, x, h, y, s_h, s_reg );
}

extern "C" __global__ void __launch_bounds__( 1024 )
    _cupy_channelizer_32x32_complex128_complex128( const int n_chans,
                                                   const int n_taps,
                                                   const int n_pts,
                                                   const thrust::complex<double> *__restrict__ x,
                                                   const thrust::complex<double> *__restrict__ h,
                                                   thrust::complex<double> *__restrict__ y ) {

    __shared__ thrust::complex<double> s_h[32][32];
    __shared__ thrust::complex<double> s_reg[32][32];

    _cupy_channelizer_32x32<thrust::complex<double>, thrust::complex<double>>(
        n_chans, n_taps, n_pts, x, h, y, s_h, s_reg );
}
