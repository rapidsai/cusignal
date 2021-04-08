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
//                                SOSFILT                                    //
///////////////////////////////////////////////////////////////////////////////

constexpr int sos_width = 6;

template<typename T>
__device__ void _cupy_sosfilt( const int n_signals,
                               const int n_samples,
                               const int n_sections,
                               const int zi_width,
                               const T *__restrict__ sos,
                               const T *__restrict__ zi,
                               T *__restrict__ x_in,
                               T *s_buffer ) {

    T *s_out { s_buffer };
    T *s_zi { reinterpret_cast<T *>( &s_out[n_sections] ) };
    T *s_sos { reinterpret_cast<T *>( &s_zi[n_sections * zi_width] ) };

    const int tx { static_cast<int>( threadIdx.x ) };
    const int ty { static_cast<int>( blockIdx.y * blockDim.y + threadIdx.y ) };

    // Reset shared memory
    s_out[tx] = 0;

    // Load zi
    for ( int i = 0; i < zi_width; i++ ) {
        s_zi[tx * zi_width + i] = zi[ty * n_sections * zi_width + tx * zi_width + i];
    }

    // Load SOS
    // b is in s_sos[tx * sos_width + [0-2]]
    // a is in s_sos[tx * sos_width + [3-5]]
#pragma unroll sos_width
    for ( int i = 0; i < sos_width; i++ ) {
        s_sos[tx * sos_width + i] = sos[tx * sos_width + i];
    }

    __syncthreads( );

    const int load_size { n_sections - 1 };
    const int unload_size { n_samples - load_size };

    T temp {};
    T x_n {};

    if ( ty < n_signals ) {
        // Loading phase
        for ( int n = 0; n < load_size; n++ ) {
            if ( tx == 0 ) {
                x_n = x_in[ty * n_samples + n];
            } else {
                x_n = s_out[tx - 1];
            }

            // Use direct II transposed structure
            temp = s_sos[tx * sos_width + 0] * x_n + s_zi[tx * zi_width + 0];

            s_zi[tx * zi_width + 0] =
                s_sos[tx * sos_width + 1] * x_n - s_sos[tx * sos_width + 4] * temp + s_zi[tx * zi_width + 1];

            s_zi[tx * zi_width + 1] = s_sos[tx * sos_width + 2] * x_n - s_sos[tx * sos_width + 5] * temp;

            s_out[tx] = temp;

            __syncthreads( );
        }

        // Processing phase
        for ( int n = load_size; n < n_samples; n++ ) {
            if ( tx == 0 ) {
                x_n = x_in[ty * n_samples + n];
            } else {
                x_n = s_out[tx - 1];
            }

            // Use direct II transposed structure
            temp = s_sos[tx * sos_width + 0] * x_n + s_zi[tx * zi_width + 0];

            s_zi[tx * zi_width + 0] =
                s_sos[tx * sos_width + 1] * x_n - s_sos[tx * sos_width + 4] * temp + s_zi[tx * zi_width + 1];

            s_zi[tx * zi_width + 1] = s_sos[tx * sos_width + 2] * x_n - s_sos[tx * sos_width + 5] * temp;

            if ( tx < load_size ) {
                s_out[tx] = temp;
            } else {
                x_in[ty * n_samples + ( n - load_size )] = temp;
            }

            __syncthreads( );
        }

        // Unloading phase
        for ( int n = 0; n < n_sections; n++ ) {
            // retire threads that are less than n
            if ( tx > n ) {
                x_n = s_out[tx - 1];

                // Use direct II transposed structure
                temp = s_sos[tx * sos_width + 0] * x_n + s_zi[tx * zi_width + 0];

                s_zi[tx * zi_width + 0] =
                    s_sos[tx * sos_width + 1] * x_n - s_sos[tx * sos_width + 4] * temp + s_zi[tx * zi_width + 1];

                s_zi[tx * zi_width + 1] = s_sos[tx * sos_width + 2] * x_n - s_sos[tx * sos_width + 5] * temp;

                if ( tx < load_size ) {
                    s_out[tx] = temp;
                } else {
                    x_in[ty * n_samples + ( n + unload_size )] = temp;
                }
                __syncthreads( );
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__( 1024 ) _cupy_sosfilt_float32( const int n_signals,
                                                                            const int n_samples,
                                                                            const int n_sections,
                                                                            const int zi_width,
                                                                            const float *__restrict__ sos,
                                                                            const float *__restrict__ zi,
                                                                            float *__restrict__ x_in ) {

    extern __shared__ float s_buffer_f[];

    _cupy_sosfilt<float>( n_signals, n_samples, n_sections, zi_width, sos, zi, x_in, s_buffer_f );
}

extern "C" __global__ void __launch_bounds__( 1024 ) _cupy_sosfilt_float64( const int n_signals,
                                                                            const int n_samples,
                                                                            const int n_sections,
                                                                            const int zi_width,
                                                                            const double *__restrict__ sos,
                                                                            const double *__restrict__ zi,
                                                                            double *__restrict__ x_in ) {

    extern __shared__ double s_buffer_d[];

    _cupy_sosfilt<double>( n_signals, n_samples, n_sections, zi_width, sos, zi, x_in, s_buffer_d );
}
