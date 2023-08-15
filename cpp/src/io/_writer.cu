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
//                                WRITER                                     //
///////////////////////////////////////////////////////////////////////////////

template<typename T>
__device__ void _cupy_pack( const size_t N, T *__restrict__ input, unsigned char *__restrict__ output ) {

    const int tx { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
    const int stride { static_cast<int>( blockDim.x * gridDim.x ) };

    for ( int tid = tx; tid < N; tid += stride ) {
        output[tid] = reinterpret_cast<unsigned char *>( input )[tid];
    }
}

extern "C" __global__ void __launch_bounds__( 512 )
    _cupy_pack_int8( const size_t N, char *__restrict__ input, unsigned char *__restrict__ output ) {
    _cupy_pack<char>( N, input, output );
}

extern "C" __global__ void __launch_bounds__( 512 )
    _cupy_pack_uint8( const size_t N, unsigned char *__restrict__ input, unsigned char *__restrict__ output ) {
    _cupy_pack<unsigned char>( N, input, output );
}

extern "C" __global__ void __launch_bounds__( 512 )
    _cupy_pack_int16( const size_t N, short *__restrict__ input, unsigned char *__restrict__ output ) {
    _cupy_pack<short>( N, input, output );
}

extern "C" __global__ void __launch_bounds__( 512 )
    _cupy_pack_uint16( const size_t N, unsigned short *__restrict__ input, unsigned char *__restrict__ output ) {
    _cupy_pack<unsigned short>( N, input, output );
}

extern "C" __global__ void __launch_bounds__( 512 )
    _cupy_pack_int32( const size_t N, int *__restrict__ input, unsigned char *__restrict__ output ) {
    _cupy_pack<int>( N, input, output );
}

extern "C" __global__ void __launch_bounds__( 512 )
    _cupy_pack_uint32( const size_t N, unsigned int *__restrict__ input, unsigned char *__restrict__ output ) {
    _cupy_pack<unsigned int>( N, input, output );
}

extern "C" __global__ void __launch_bounds__( 512 )
    _cupy_pack_float32( const size_t N, float *__restrict__ input, unsigned char *__restrict__ output ) {
    _cupy_pack<float>( N, input, output );
}

extern "C" __global__ void __launch_bounds__( 512 )
    _cupy_pack_float64( const size_t N, double *__restrict__ input, unsigned char *__restrict__ output ) {
    _cupy_pack<double>( N, input, output );
}

extern "C" __global__ void __launch_bounds__( 512 ) _cupy_pack_complex64( const size_t N,
                                                                          thrust::complex<float> *__restrict__ input,
                                                                          unsigned char *__restrict__ output ) {
    _cupy_pack<thrust::complex<float>>( N, input, output );
}

extern "C" __global__ void __launch_bounds__( 512 ) _cupy_pack_complex128( const size_t N,
                                                                           thrust::complex<double> *__restrict__ input,
                                                                           unsigned char *__restrict__ output ) {
    _cupy_pack<thrust::complex<double>>( N, input, output );
}
