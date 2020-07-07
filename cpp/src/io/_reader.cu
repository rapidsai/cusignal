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
//                                READER                                     //
///////////////////////////////////////////////////////////////////////////////

// Byte swap short
__device__ short swap_int16( short val )
{
	return (val << 8) | ((val >> 8) & 0xFF);
}

// Byte swap unsigned short
__device__ unsigned short swap_uint16( unsigned short val )
{
	return (val << 8) | (val >> 8 );
}

// Byte swap int
__device__ int swap_int32( int val )
{
	val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF );
	return (val << 16) | ((val >> 16) & 0xFFFF);
}

// Byte swap unsigned int
__device__ unsigned int swap_uint32( unsigned int val )
{
	val = ((val << 8) & 0xFF00FF00 ) | ((val >> 8) & 0xFF00FF );
	return (val << 16) | (val >> 16);
}

// Byte swap float
__device__ float swap_float( float val )
{
	float retVal;
	char *floatToConvert = reinterpret_cast<char*>(&val);
	char *returnFloat = reinterpret_cast<char*>(&retVal);

	int ds = sizeof(float); // data size

	// swap the bytes into a temporary buffer
	#pragma unroll 4
	for ( int i = 0; i < ds; i++ ) {
		returnFloat[i] = floatToConvert[(ds - 1) - i];
	}

	return retVal;
}

__device__ double swap_double( double val )
{
	double retVal;
	char *doubleToConvert = reinterpret_cast<char*>(&val);
	char *returnDouble = reinterpret_cast<char*>(&retVal);

	int ds = sizeof(double); // data size

	// swap the bytes into a temporary buffer
	#pragma unroll 8
	for ( int i = 0; i < ds; i++ ) {
		returnDouble[i] = doubleToConvert[(ds - 1) - i];
	}

	return retVal;
}

template<typename T>
__device__ void _cupy_unpack(
		const size_t N,
		const bool little,
		unsigned char * __restrict__ input,
		T * __restrict__ output) {

	const int tx {
		static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) };
	const int stride { static_cast<int>(blockDim.x * gridDim.x) };

	for ( int tid = tx; tid < N; tid += stride ) {

		if ( little ) {
			output[tid] = reinterpret_cast<T*>(input)[tid];
		} else {
			T data = reinterpret_cast<T*>(input)[tid];

			if (std::is_same<T, char>::value) {
				output[tid] = data;
			}
			else if (std::is_same<T, short>::value) {
				T temp = swap_int16(data);
				output[tid] = temp;
			}
			else if  (std::is_same<T, unsigned short>::value) {
				T temp = swap_uint16(data);
				output[tid] = temp;
			}
			else if  (std::is_same<T, int>::value) {
				T temp = swap_int32(data);
				output[tid] = temp;
			}
			else if  (std::is_same<T, unsigned int>::value) {
				T temp = swap_uint32(data);
				output[tid] = temp;
			}
			else if  (std::is_same<T, float>::value) {
				T temp = swap_float(data);
				output[tid] = temp;
			}
			else if  (std::is_same<T, double>::value) {
				T temp = swap_double(data);
				output[tid] = temp;
			}
		}
	}
}

template<typename T>
__device__ void _cupy_unpack_complex(
		const size_t N,
		const bool little,
		unsigned char * __restrict__ input,
		T * __restrict__ output) {

	const int tx {
		static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) };
	const int stride { static_cast<int>(blockDim.x * gridDim.x) };

	for ( int tid = tx; tid < N; tid += stride ) {

		if ( little ) {
			output[tid] = reinterpret_cast<T*>(input)[tid];
		} else {
			T data = reinterpret_cast<T*>(input)[tid];

			if  (std::is_same<T, thrust::complex<float>>::value) {
				float real = swap_float(data.real());
				float imag = swap_float(data.imag());

				output[tid] = thrust::complex<float>(real, imag);
			}
			else if (std::is_same<T, thrust::complex<double>>::value) {
				double real = swap_double(data.real());
				double imag = swap_double(data.imag());

				output[tid] = thrust::complex<double>(real, imag);
			}
		}
	}
}

extern "C" __global__ void _cupy_unpack_int8(
	const size_t N,
	const bool little,
	unsigned char * __restrict__ input,
	char * __restrict__ output
	) {
	_cupy_unpack<char>(N, little, input, output);
}

extern "C" __global__ void _cupy_unpack_uint8(
	const size_t N,
	const bool little,
	unsigned char * __restrict__ input,
	unsigned char * __restrict__ output
	) {
	_cupy_unpack<unsigned char>(N, little, input, output);
}

extern "C" __global__ void _cupy_unpack_int16(
	const size_t N,
	const bool little,
	unsigned char * __restrict__ input,
	short * __restrict__ output
	) {
	_cupy_unpack<short>(N, little, input, output);
}

extern "C" __global__ void _cupy_unpack_uint16(
	const size_t N,
	const bool little,
	unsigned char * __restrict__ input,
	unsigned short * __restrict__ output
	) {
	_cupy_unpack<unsigned short>(N, little, input, output);
}

extern "C" __global__ void _cupy_unpack_int32(
	const size_t N,
	const bool little,
	unsigned char * __restrict__ input,
	int * __restrict__ output
	) {
	_cupy_unpack<int>(N, little, input, output);
}

extern "C" __global__ void _cupy_unpack_uint32(
	const size_t N,
	const bool little,
	unsigned char * __restrict__ input,
	unsigned int * __restrict__ output
	) {
	_cupy_unpack<unsigned int>(N, little, input, output);
}

extern "C" __global__ void _cupy_unpack_float32(
	const size_t N,
	const bool little,
	unsigned char * __restrict__ input,
	float * __restrict__ output
	) {
	_cupy_unpack<float>(N, little, input, output);
}

extern "C" __global__ void _cupy_unpack_float64(
	const size_t N,
	const bool little,
	unsigned char * __restrict__ input,
	double * __restrict__ output
	) {
	_cupy_unpack<double>(N, little, input, output);
}

extern "C" __global__ void _cupy_unpack_complex64(
	const size_t N,
	const bool little,
	unsigned char * __restrict__ input,
	thrust::complex<float> * __restrict__ output
	) {
	_cupy_unpack_complex<thrust::complex<float>>(N, little, input, output);
}

extern "C" __global__ void _cupy_unpack_complex128(
	const size_t N,
	const bool little,
	unsigned char * __restrict__ input,
	thrust::complex<double> * __restrict__ output
	) {
	_cupy_unpack_complex<thrust::complex<double>>(N, little, input, output);
}
