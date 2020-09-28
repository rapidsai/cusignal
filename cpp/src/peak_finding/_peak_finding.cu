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

template<typename T>
__device__ void
_cupy_boolrelextrema( const int n, const int order, const T *__restrict__ inp, T *__restrict__ results ) {

    const int tx { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
    const int stride { static_cast<int>( blockDim.x * gridDim.x ) };

    for ( int tid = tx; tid < n; tid += stride ) {

        T temp {};

        results[tid] = temp;
    }
}

extern "C" __global__ void __launch_bounds__( 512 )
    _cupy_boolrelextrema_int32( const int n, const int order, const int *__restrict__ inp, int *__restrict__ results ) {
    _cupy_boolrelextrema<int>( n, order, inp, results );
}

extern "C" __global__ void __launch_bounds__( 512 ) _cupy_boolrelextrema_int64( const int n,
                                                                                const int order,
                                                                                const long int *__restrict__ inp,
                                                                                long int *__restrict__ results ) {
    _cupy_boolrelextrema<long int>( n, order, inp, results );
}

extern "C" __global__ void __launch_bounds__( 512 ) _cupy_boolrelextrema_float32( const int n,
                                                                                  const int order,
                                                                                  const float *__restrict__ inp,
                                                                                  float *__restrict__ results ) {
    _cupy_boolrelextrema<float>( n, order, inp, results );
}

extern "C" __global__ void __launch_bounds__( 512 ) _cupy_boolrelextrema_float64( const int n,
                                                                                  const int order,
                                                                                  const double *__restrict__ inp,
                                                                                  double *__restrict__ results ) {
    _cupy_boolrelextrema<double>( n, order, inp, results );
}