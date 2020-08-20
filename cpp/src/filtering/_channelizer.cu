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
#include <cuComplex.h>

///////////////////////////////////////////////////////////////////////////////
//                            CHANNELIZER                                    //
///////////////////////////////////////////////////////////////////////////////

template<typename T>
__device__ void _cupy_channelizer( const int n_chans, const int n_taps, const int n_pts ) {

    const int ty { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
    const int tx { static_cast<int>( blockIdx.y * blockDim.y + threadIdx.y ) };

    if ( tx == 0 && ty == 0 )
        printf( "Hello\n" );
}

extern "C" __global__ void __launch_bounds__( 64 )
    _cupy_channelizer_float32( const int n_chans, const int n_taps, const int n_pts ) {
    _cupy_channelizer<float>( n_chans, n_taps, n_pts );
}

extern "C" __global__ void __launch_bounds__( 64 )
    _cupy_channelizer_float64( const int n_chans, const int n_taps, const int n_pts ) {
    _cupy_channelizer<double>( n_chans, n_taps, n_pts );
}

extern "C" __global__ void __launch_bounds__( 64 )
    _cupy_channelizer_complex64( const int n_chans, const int n_taps, const int n_pts ) {
    _cupy_channelizer<cuFloatComplex>( n_chans, n_taps, n_pts );
}

extern "C" __global__ void __launch_bounds__( 64 )
    _cupy_channelizer_complex128( const int n_chans, const int n_taps, const int n_pts ) {
    _cupy_channelizer<cuDoubleComplex>( n_chans, n_taps, n_pts );
}

///////////////////////////////////////////////////////////////////////////////
//                          CHANNELIZER 8x8                                  //
///////////////////////////////////////////////////////////////////////////////

template<typename T>
__device__ void _cupy_channelizer_8x8( const int n_chans, const int n_taps, const int n_pts, T s_h[8][8], T s_reg[8][8] ) {

    const int ty { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
	const int tx { static_cast<int>( blockIdx.y * blockDim.y + threadIdx.y ) };
	
	if ( tx == 0 && ty == 0 )
        printf( "Hello from 8x8\n" );
}

extern "C" __global__ void __launch_bounds__( 64 )
    _cupy_channelizer_8x8_float32( const int n_chans, const int n_taps, const int n_pts ) {

    __shared__ float s_h[8][8];
    __shared__ float s_reg[8][8];

    _cupy_channelizer_8x8<float>( n_chans, n_taps, n_pts, s_h, s_reg );
}

extern "C" __global__ void __launch_bounds__( 64 )
    _cupy_channelizer_8x8_float64( const int n_chans, const int n_taps, const int n_pts ) {

    __shared__ double s_h[8][8];
    __shared__ double s_reg[8][8];

    _cupy_channelizer_8x8<double>( n_chans, n_taps, n_pts, s_h, s_reg );
}

extern "C" __global__ void __launch_bounds__( 64 )
    _cupy_channelizer_8x8_complex64( const int n_chans, const int n_taps, const int n_pts ) {

    __shared__ cuFloatComplex s_h[8][8];
    __shared__ cuFloatComplex s_reg[8][8];

    _cupy_channelizer_8x8<cuFloatComplex>( n_chans, n_taps, n_pts, s_h, s_reg );
}

extern "C" __global__ void __launch_bounds__( 64 )
    _cupy_channelizer_8x8_complex128( const int n_chans, const int n_taps, const int n_pts ) {

    __shared__ cuDoubleComplex s_h[8][8];
    __shared__ cuDoubleComplex s_reg[8][8];

    _cupy_channelizer_8x8<cuDoubleComplex>( n_chans, n_taps, n_pts, s_h, s_reg );
}

///////////////////////////////////////////////////////////////////////////////
//                        CHANNELIZER 16x16                                  //
///////////////////////////////////////////////////////////////////////////////

template<typename T>
__device__ void _cupy_channelizer_16x16( const int n_chans, const int n_taps, const int n_pts, T s_h[16][16], T s_reg[16][16] ) {

    const int ty { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
	const int tx { static_cast<int>( blockIdx.y * blockDim.y + threadIdx.y ) };
	
	if ( tx == 0 && ty == 0 )
        printf( "Hello from 16x16\n" );
}

extern "C" __global__ void __launch_bounds__( 256 )
    _cupy_channelizer_16x16_float32( const int n_chans, const int n_taps, const int n_pts ) {

    __shared__ float s_h[16][16];
    __shared__ float s_reg[16][16];

    _cupy_channelizer_16x16<float>( n_chans, n_taps, n_pts, s_h, s_reg );
}

extern "C" __global__ void __launch_bounds__( 256 )
    _cupy_channelizer_16x16_float64( const int n_chans, const int n_taps, const int n_pts ) {

    __shared__ double s_h[16][16];
    __shared__ double s_reg[16][16];

    _cupy_channelizer_16x16<double>( n_chans, n_taps, n_pts, s_h, s_reg );
}

extern "C" __global__ void __launch_bounds__( 256 )
    _cupy_channelizer_16x16_complex64( const int n_chans, const int n_taps, const int n_pts ) {

    __shared__ cuFloatComplex s_h[16][16];
    __shared__ cuFloatComplex s_reg[16][16];

    _cupy_channelizer_16x16<cuFloatComplex>( n_chans, n_taps, n_pts, s_h, s_reg );
}

extern "C" __global__ void __launch_bounds__( 256 )
    _cupy_channelizer_16x16_complex128( const int n_chans, const int n_taps, const int n_pts ) {

    __shared__ cuDoubleComplex s_h[16][16];
    __shared__ cuDoubleComplex s_reg[16][16];

    _cupy_channelizer_16x16<cuDoubleComplex>( n_chans, n_taps, n_pts, s_h, s_reg );
}

///////////////////////////////////////////////////////////////////////////////
//                        CHANNELIZER 32x32                                  //
///////////////////////////////////////////////////////////////////////////////

template<typename T>
__device__ void _cupy_channelizer_32x32( const int n_chans, const int n_taps, const int n_pts, T s_h[32][32], T s_reg[32][32] ) {

    const int ty { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
	const int tx { static_cast<int>( blockIdx.y * blockDim.y + threadIdx.y ) };
	
	if ( tx == 0 && ty == 0 )
        printf( "Hello from 32x32\n" );
}

extern "C" __global__ void __launch_bounds__( 1024 )
    _cupy_channelizer_32x32_float32( const int n_chans, const int n_taps, const int n_pts ) {

    __shared__ float s_h[32][32];
    __shared__ float s_reg[32][32];

    _cupy_channelizer_32x32<float>( n_chans, n_taps, n_pts, s_h, s_reg );
}

extern "C" __global__ void __launch_bounds__( 1024 )
    _cupy_channelizer_32x32_float64( const int n_chans, const int n_taps, const int n_pts ) {

    __shared__ double s_h[32][32];
    __shared__ double s_reg[32][32];

    _cupy_channelizer_32x32<double>( n_chans, n_taps, n_pts, s_h, s_reg );
}

extern "C" __global__ void __launch_bounds__( 1024 )
    _cupy_channelizer_32x32_complex64( const int n_chans, const int n_taps, const int n_pts ) {

    __shared__ cuFloatComplex s_h[32][32];
    __shared__ cuFloatComplex s_reg[32][32];

    _cupy_channelizer_32x32<cuFloatComplex>( n_chans, n_taps, n_pts, s_h, s_reg );
}

extern "C" __global__ void __launch_bounds__( 1024 )
    _cupy_channelizer_32x32_complex128( const int n_chans, const int n_taps, const int n_pts ) {

    __shared__ cuDoubleComplex s_h[32][32];
    __shared__ cuDoubleComplex s_reg[32][32];

    _cupy_channelizer_32x32<cuDoubleComplex>( n_chans, n_taps, n_pts, s_h, s_reg );
}
