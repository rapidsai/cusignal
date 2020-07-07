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
//                                WRITER                                     //
///////////////////////////////////////////////////////////////////////////////

template<typename T>
__device__ void _cupy_pack(
		const size_t N,
		T * __restrict__ input,
		unsigned char * __restrict__ output) {

	const int tx {
		static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) };
	const int stride { static_cast<int>(blockDim.x * gridDim.x) };

	for ( int tid = tx; tid < N; tid += stride ) {
		output[tid] = reinterpret_cast<unsigned char*>(input)[tid];
	}
}

extern "C" __global__ void _cupy_pack_int8(
	const size_t N,
	char * __restrict__ input,
	unsigned char * __restrict__ output
	) {
	_cupy_pack<char>(N, input, output);
}

extern "C" __global__ void _cupy_pack_uint8(
	const size_t N,
	unsigned char * __restrict__ input,
	unsigned char * __restrict__ output
	) {
	_cupy_pack<unsigned char>(N, input, output);
}

extern "C" __global__ void _cupy_pack_int16(
	const size_t N,
	short * __restrict__ input,
	unsigned char * __restrict__ output
	) {
	_cupy_pack<short>(N, input, output);
}

extern "C" __global__ void _cupy_pack_uint16(
	const size_t N,
	unsigned short * __restrict__ input,
	unsigned char * __restrict__ output
	) {
	_cupy_pack<unsigned short>(N, input, output);
}

extern "C" __global__ void _cupy_pack_int32(
	const size_t N,
	int * __restrict__ input,
	unsigned char * __restrict__ output
	) {
	_cupy_pack<int>(N, input, output);
}

extern "C" __global__ void _cupy_pack_uint32(
	const size_t N,
	unsigned int * __restrict__ input,
	unsigned char * __restrict__ output
	) {
	_cupy_pack<unsigned int>(N, input, output);
}

extern "C" __global__ void _cupy_pack_float32(
	const size_t N,
	float * __restrict__ input,
	unsigned char * __restrict__ output
	) {
	_cupy_pack<float>(N, input, output);
}

extern "C" __global__ void _cupy_pack_float64(
	const size_t N,
	double * __restrict__ input,
	unsigned char * __restrict__ output
	) {
	_cupy_pack<double>(N, input, output);
}

extern "C" __global__ void _cupy_pack_complex64(
	const size_t N,
	thrust::complex<float> * __restrict__ input,
	unsigned char * __restrict__ output
	) {
	_cupy_pack<thrust::complex<float>>(N, input, output);
}

extern "C" __global__ void _cupy_pack_complex128(
	const size_t N,
	thrust::complex<double> * __restrict__ input,
	unsigned char * __restrict__ output
	) {
	_cupy_pack<thrust::complex<double>>(N, input, output);
}
