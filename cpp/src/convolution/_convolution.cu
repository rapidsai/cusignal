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
//                              CONVOLVE                                     //
///////////////////////////////////////////////////////////////////////////////

template<typename T>
__device__ void _cupy_convolve( const T *__restrict__ inp,
                                const int inpW,
                                const T *__restrict__ kernel,
                                const int  kerW,
                                const int  mode,
                                const bool swapped_inputs,
                                T *__restrict__ out,
                                const int outW ) {

    const int tx { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
    const int stride { static_cast<int>( blockDim.x * gridDim.x ) };

    for ( int tid = tx; tid < outW; tid += stride ) {

        T temp {};

        if ( mode == 0 ) {  // Valid
            if ( tid >= 0 && tid < inpW ) {
                for ( int j = 0; j < kerW; j++ ) {
                    temp += inp[tid + j] * kernel[( kerW - 1 ) - j];
                }
            }
        } else if ( mode == 1 ) {  // Same
            const int P1 { kerW / 2 };
            int       start {};
            if ( !swapped_inputs ) {
                start = 0 - P1 + tid;
            } else {
                start = ( ( inpW - 1 ) / 2 ) - ( kerW - 1 ) + tid;
            }
            for ( int j = 0; j < kerW; j++ ) {
                if ( ( start + j >= 0 ) && ( start + j < inpW ) ) {
                    temp += inp[start + j] * kernel[( kerW - 1 ) - j];
                }
            }
        } else {  // Full
            const int P1 { kerW - 1 };
            const int start { 0 - P1 + tid };
            for ( int j = 0; j < kerW; j++ ) {
                if ( ( start + j >= 0 ) && ( start + j < inpW ) ) {
                    temp += inp[start + j] * kernel[( kerW - 1 ) - j];
                }
            }
        }

        out[tid] = temp;
    }
}

extern "C" __global__ void __launch_bounds__( 512 ) _cupy_convolve_int32( const int *__restrict__ inp,
                                                                          const int inpW,
                                                                          const int *__restrict__ kernel,
                                                                          const int  kerW,
                                                                          const int  mode,
                                                                          const bool swapped_inputs,
                                                                          int *__restrict__ out,
                                                                          const int outW ) {
    _cupy_convolve<int>( inp, inpW, kernel, kerW, mode, swapped_inputs, out, outW );
}

extern "C" __global__ void __launch_bounds__( 512 ) _cupy_convolve_int64( const long int *__restrict__ inp,
                                                                          const int inpW,
                                                                          const long int *__restrict__ kernel,
                                                                          const int  kerW,
                                                                          const int  mode,
                                                                          const bool swapped_inputs,
                                                                          long int *__restrict__ out,
                                                                          const int outW ) {
    _cupy_convolve<long int>( inp, inpW, kernel, kerW, mode, swapped_inputs, out, outW );
}

extern "C" __global__ void __launch_bounds__( 512 ) _cupy_convolve_float32( const float *__restrict__ inp,
                                                                            const int inpW,
                                                                            const float *__restrict__ kernel,
                                                                            const int  kerW,
                                                                            const int  mode,
                                                                            const bool swapped_inputs,
                                                                            float *__restrict__ out,
                                                                            const int outW ) {
    _cupy_convolve<float>( inp, inpW, kernel, kerW, mode, swapped_inputs, out, outW );
}

extern "C" __global__ void __launch_bounds__( 512 ) _cupy_convolve_float64( const double *__restrict__ inp,
                                                                            const int inpW,
                                                                            const double *__restrict__ kernel,
                                                                            const int  kerW,
                                                                            const int  mode,
                                                                            const bool swapped_inputs,
                                                                            double *__restrict__ out,
                                                                            const int outW ) {
    _cupy_convolve<double>( inp, inpW, kernel, kerW, mode, swapped_inputs, out, outW );
}

extern "C" __global__ void __launch_bounds__( 512 )
    _cupy_convolve_complex64( thrust::complex<float> *__restrict__ inp,
                              const int inpW,
                              thrust::complex<float> *__restrict__ kernel,
                              const int  kerW,
                              const int  mode,
                              const bool swapped_inputs,
                              thrust::complex<float> *__restrict__ out,
                              const int outW ) {
    _cupy_convolve<thrust::complex<float>>( inp, inpW, kernel, kerW, mode, swapped_inputs, out, outW );
}

extern "C" __global__ void __launch_bounds__( 512 )
    _cupy_convolve_complex128( const thrust::complex<double> *__restrict__ inp,
                               const int inpW,
                               const thrust::complex<double> *__restrict__ kernel,
                               const int  kerW,
                               const int  mode,
                               const bool swapped_inputs,
                               thrust::complex<double> *__restrict__ out,
                               const int outW ) {
    _cupy_convolve<thrust::complex<double>>( inp, inpW, kernel, kerW, mode, swapped_inputs, out, outW );
}

///////////////////////////////////////////////////////////////////////////////
//                              CORRELATE                                    //
///////////////////////////////////////////////////////////////////////////////

template<typename T>
__device__ void _cupy_correlate( const T *__restrict__ inp,
                                 const int inpW,
                                 const T *__restrict__ kernel,
                                 const int  kerW,
                                 const int  mode,
                                 const bool swapped_inputs,
                                 T *__restrict__ out,
                                 const int outW ) {

    const int tx { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
    const int stride { static_cast<int>( blockDim.x * gridDim.x ) };

    for ( int tid = tx; tid < outW; tid += stride ) {
        T temp {};

        if ( mode == 0 ) {  // Valid
            if ( tid >= 0 && tid < inpW ) {
                for ( int j = 0; j < kerW; j++ ) {
                    temp += inp[tid + j] * kernel[j];
                }
            }
        } else if ( mode == 1 ) {  // Same
            const int P1 { kerW / 2 };
            int       start {};
            if ( !swapped_inputs ) {
                start = 0 - P1 + tid;
            } else {
                start = ( ( inpW - 1 ) / 2 ) - ( kerW - 1 ) + tid + 1;
            }
            for ( int j = 0; j < kerW; j++ ) {
                if ( ( start + j >= 0 ) && ( start + j < inpW ) ) {
                    temp += inp[start + j] * kernel[j];
                }
            }
        } else {  // Full
            const int P1 { kerW - 1 };
            const int start { 0 - P1 + tid };
            for ( int j = 0; j < kerW; j++ ) {
                if ( ( start + j >= 0 ) && ( start + j < inpW ) ) {
                    temp += inp[start + j] * kernel[j];
                }
            }
        }

        if ( swapped_inputs ) {
            out[outW - tid - 1] = temp;  // TODO: Move to shared memory
        } else {
            out[tid] = temp;
        }
    }
}

extern "C" __global__ void __launch_bounds__( 512 ) _cupy_correlate_int32( const int *__restrict__ inp,
                                                                           const int inpW,
                                                                           const int *__restrict__ kernel,
                                                                           const int  kerW,
                                                                           const int  mode,
                                                                           const bool swapped_inputs,
                                                                           int *__restrict__ out,
                                                                           const int outW ) {
    _cupy_correlate<int>( inp, inpW, kernel, kerW, mode, swapped_inputs, out, outW );
}

extern "C" __global__ void __launch_bounds__( 512 ) _cupy_correlate_int64( const long int *__restrict__ inp,
                                                                           const int inpW,
                                                                           const long int *__restrict__ kernel,
                                                                           const int  kerW,
                                                                           const int  mode,
                                                                           const bool swapped_inputs,
                                                                           long int *__restrict__ out,
                                                                           const int outW ) {
    _cupy_correlate<long int>( inp, inpW, kernel, kerW, mode, swapped_inputs, out, outW );
}

extern "C" __global__ void __launch_bounds__( 512 ) _cupy_correlate_float32( const float *__restrict__ inp,
                                                                             const int inpW,
                                                                             const float *__restrict__ kernel,
                                                                             const int  kerW,
                                                                             const int  mode,
                                                                             const bool swapped_inputs,
                                                                             float *__restrict__ out,
                                                                             const int outW ) {
    _cupy_correlate<float>( inp, inpW, kernel, kerW, mode, swapped_inputs, out, outW );
}

extern "C" __global__ void __launch_bounds__( 512 ) _cupy_correlate_float64( const double *__restrict__ inp,
                                                                             const int inpW,
                                                                             const double *__restrict__ kernel,
                                                                             const int  kerW,
                                                                             const int  mode,
                                                                             const bool swapped_inputs,
                                                                             double *__restrict__ out,
                                                                             const int outW ) {
    _cupy_correlate<double>( inp, inpW, kernel, kerW, mode, swapped_inputs, out, outW );
}

extern "C" __global__ void __launch_bounds__( 512 )
    _cupy_correlate_complex64( thrust::complex<float> *__restrict__ inp,
                               const int inpW,
                               thrust::complex<float> *__restrict__ kernel,
                               const int  kerW,
                               const int  mode,
                               const bool swapped_inputs,
                               thrust::complex<float> *__restrict__ out,
                               const int outW ) {
    _cupy_correlate<thrust::complex<float>>( inp, inpW, kernel, kerW, mode, swapped_inputs, out, outW );
}

extern "C" __global__ void __launch_bounds__( 512 )
    _cupy_correlate_complex128( const thrust::complex<double> *__restrict__ inp,
                                const int inpW,
                                const thrust::complex<double> *__restrict__ kernel,
                                const int  kerW,
                                const int  mode,
                                const bool swapped_inputs,
                                thrust::complex<double> *__restrict__ out,
                                const int outW ) {
    _cupy_correlate<thrust::complex<double>>( inp, inpW, kernel, kerW, mode, swapped_inputs, out, outW );
}

///////////////////////////////////////////////////////////////////////////////
//                              CONVOLVE 2D                                  //
///////////////////////////////////////////////////////////////////////////////

template<typename T>
__device__ void _cupy_convolve2D( const T *__restrict__ inp,
                                  const int inpW,
                                  const int inpH,
                                  const T *__restrict__ kernel,
                                  const int kerW,
                                  const int kerH,
                                  const int S0,
                                  const int S1,
                                  T *__restrict__ out,
                                  const int outW,
                                  const int outH,
                                  const int pick ) {

    const int ty { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
    const int tx { static_cast<int>( blockIdx.y * blockDim.y + threadIdx.y ) };

    int i {};
    if ( pick != 3 ) {
        i = tx + S0;
    } else {
        i = tx + S1;
    }
    int j { ty + S0 };

    int2 oPixelPos { tx, ty };
    if ( ( tx < outH ) && ( ty < outW ) ) {
        T temp {};

        // Odd kernel
        if ( pick == 1 ) {
            for ( int k = -S0; k < ( S0 + 1 ); k++ ) {
                for ( int l = -S0; l < ( S0 + 1 ); l++ ) {
                    int2 iPixelPos { ( i + k ), ( j + l ) };
                    int2 coefPos { ( -k + S0 ), ( -l + S0 ) };
                    temp += inp[iPixelPos.x * inpW + iPixelPos.y] * kernel[coefPos.x * kerW + coefPos.y];
                }
            }
            // Even kernel
        } else if ( pick == 2 ) {
            for ( int k = -S0; k < S0; k++ ) {
                for ( int l = -S0; l < S0; l++ ) {
                    int2 iPixelPos { ( i + k ), ( j + l ) };
                    int2 coefPos { ( -k + S0 - 1 ), ( -l + S0 - 1 ) };
                    temp += inp[iPixelPos.x * inpW + iPixelPos.y] * kernel[coefPos.x * kerW + coefPos.y];
                }
            }

            // Non-squares kernel
        } else {
            for ( int k = 0; k < S0; k++ ) {
                for ( int l = 0; l < S1; l++ ) {
                    int2 iPixelPos { ( i + k - S1 ), ( j + l - S0 ) };
                    int2 coefPos { ( -k + S0 - 1 ), ( -l + S1 - 1 ) };
                    temp += inp[iPixelPos.x * inpW + iPixelPos.y] * kernel[coefPos.x * kerH + coefPos.y];
                }
            }
        }
        out[oPixelPos.x * outW + oPixelPos.y] = temp;
    }
}

extern "C" __global__ void __launch_bounds__( 256 ) _cupy_convolve2D_int32( const int *__restrict__ inp,
                                                                            const int inpW,
                                                                            const int inpH,
                                                                            const int *__restrict__ kernel,
                                                                            const int kerW,
                                                                            const int kerH,
                                                                            const int S0,
                                                                            const int S1,
                                                                            int *__restrict__ out,
                                                                            const int outW,
                                                                            const int outH,
                                                                            const int pick ) {
    _cupy_convolve2D<int>( inp, inpW, inpH, kernel, kerW, kerH, S0, S1, out, outW, outH, pick );
}

extern "C" __global__ void __launch_bounds__( 256 ) _cupy_convolve2D_int64( const long int *__restrict__ inp,
                                                                            const int inpW,
                                                                            const int inpH,
                                                                            const long int *__restrict__ kernel,
                                                                            const int kerW,
                                                                            const int kerH,
                                                                            const int S0,
                                                                            const int S1,
                                                                            long int *__restrict__ out,
                                                                            const int outW,
                                                                            const int outH,
                                                                            const int pick ) {
    _cupy_convolve2D<long int>( inp, inpW, inpH, kernel, kerW, kerH, S0, S1, out, outW, outH, pick );
}

extern "C" __global__ void __launch_bounds__( 256 ) _cupy_convolve2D_float32( const float *__restrict__ inp,
                                                                              const int inpW,
                                                                              const int inpH,
                                                                              const float *__restrict__ kernel,
                                                                              const int kerW,
                                                                              const int kerH,
                                                                              const int S0,
                                                                              const int S1,
                                                                              float *__restrict__ out,
                                                                              const int outW,
                                                                              const int outH,
                                                                              const int pick ) {
    _cupy_convolve2D<float>( inp, inpW, inpH, kernel, kerW, kerH, S0, S1, out, outW, outH, pick );
}

extern "C" __global__ void __launch_bounds__( 256 ) _cupy_convolve2D_float64( const double *__restrict__ inp,
                                                                              const int inpW,
                                                                              const int inpH,
                                                                              const double *__restrict__ kernel,
                                                                              const int kerW,
                                                                              const int kerH,
                                                                              const int S0,
                                                                              const int S1,
                                                                              double *__restrict__ out,
                                                                              const int outW,
                                                                              const int outH,
                                                                              const int pick ) {
    _cupy_convolve2D<double>( inp, inpW, inpH, kernel, kerW, kerH, S0, S1, out, outW, outH, pick );
}

extern "C" __global__ void __launch_bounds__( 256 )
    _cupy_convolve2D_complex64( const thrust::complex<float> *__restrict__ inp,
                                const int inpW,
                                const int inpH,
                                const thrust::complex<float> *__restrict__ kernel,
                                const int kerW,
                                const int kerH,
                                const int S0,
                                const int S1,
                                thrust::complex<float> *__restrict__ out,
                                const int outW,
                                const int outH,
                                const int pick ) {
    _cupy_convolve2D<thrust::complex<float>>( inp, inpW, inpH, kernel, kerW, kerH, S0, S1, out, outW, outH, pick );
}

extern "C" __global__ void __launch_bounds__( 256 )
    _cupy_convolve2D_complex128( const thrust::complex<double> *__restrict__ inp,
                                 const int inpW,
                                 const int inpH,
                                 const thrust::complex<double> *__restrict__ kernel,
                                 const int kerW,
                                 const int kerH,
                                 const int S0,
                                 const int S1,
                                 thrust::complex<double> *__restrict__ out,
                                 const int outW,
                                 const int outH,
                                 const int pick ) {
    _cupy_convolve2D<thrust::complex<double>>( inp, inpW, inpH, kernel, kerW, kerH, S0, S1, out, outW, outH, pick );
}

///////////////////////////////////////////////////////////////////////////////
//                              CORRELATE 2D                                 //
///////////////////////////////////////////////////////////////////////////////

template<typename T>
__device__ void _cupy_correlate2D( const T *__restrict__ inp,
                                   const int inpW,
                                   const int inpH,
                                   const T *__restrict__ kernel,
                                   const int kerW,
                                   const int kerH,
                                   const int S0,
                                   const int S1,
                                   T *__restrict__ out,
                                   const int outW,
                                   const int outH,
                                   const int pick ) {

    const int ty { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
    const int tx { static_cast<int>( blockIdx.y * blockDim.y + threadIdx.y ) };

    int i {};
    if ( pick != 3 ) {
        i = tx + S0;
    } else {
        i = tx + S1;
    }
    int j { ty + S0 };

    int2 oPixelPos { tx, ty };
    if ( ( tx < outH ) && ( ty < outW ) ) {
        T temp {};

        // Odd
        if ( pick == 1 ) {
            for ( int k = -S0; k < ( S0 + 1 ); k++ ) {
                for ( int l = -S0; l < ( S0 + 1 ); l++ ) {
                    int2 iPixelPos { ( i + k ), ( j + l ) };
                    int2 coefPos { ( k + S0 ), ( l + S0 ) };
                    temp += inp[iPixelPos.x * inpW + iPixelPos.y] * kernel[coefPos.x * kerW + coefPos.y];
                }
            }

            // Even
        } else if ( pick == 2 ) {
            for ( int k = -S0; k < S0; k++ ) {
                for ( int l = -S0; l < S0; l++ ) {
                    int2 iPixelPos { ( i + k ), ( j + l ) };  // iPixelPos[1], [0]
                    int2 coefPos { ( k + S0 ), ( l + S0 ) };
                    temp += inp[iPixelPos.x * inpW + iPixelPos.y] * kernel[coefPos.x * kerW + coefPos.y];
                }
            }

            // Non-squares
        } else {
            for ( int k = 0; k < S0; k++ ) {
                for ( int l = 0; l < S1; l++ ) {
                    int2 iPixelPos { ( i + k - S1 ), ( j + l - S0 ) };
                    int2 coefPos { k, l };
                    temp += inp[iPixelPos.x * inpW + iPixelPos.y] * kernel[coefPos.x * kerH + coefPos.y];
                }
            }
        }
        out[oPixelPos.x * outW + oPixelPos.y] = temp;
    }
}

extern "C" __global__ void __launch_bounds__( 256 ) _cupy_correlate2D_int32( const int *__restrict__ inp,
                                                                             const int inpW,
                                                                             const int inpH,
                                                                             const int *__restrict__ kernel,
                                                                             const int kerW,
                                                                             const int kerH,
                                                                             const int S0,
                                                                             const int S1,
                                                                             int *__restrict__ out,
                                                                             const int outW,
                                                                             const int outH,
                                                                             const int pick ) {
    _cupy_correlate2D<int>( inp, inpW, inpH, kernel, kerW, kerH, S0, S1, out, outW, outH, pick );
}

extern "C" __global__ void __launch_bounds__( 256 ) _cupy_correlate2D_int64( const long int *__restrict__ inp,
                                                                             const int inpW,
                                                                             const int inpH,
                                                                             const long int *__restrict__ kernel,
                                                                             const int kerW,
                                                                             const int kerH,
                                                                             const int S0,
                                                                             const int S1,
                                                                             long int *__restrict__ out,
                                                                             const int outW,
                                                                             const int outH,
                                                                             const int pick ) {
    _cupy_correlate2D<long int>( inp, inpW, inpH, kernel, kerW, kerH, S0, S1, out, outW, outH, pick );
}

extern "C" __global__ void __launch_bounds__( 256 ) _cupy_correlate2D_float32( const float *__restrict__ inp,
                                                                               const int inpW,
                                                                               const int inpH,
                                                                               const float *__restrict__ kernel,
                                                                               const int kerW,
                                                                               const int kerH,
                                                                               const int S0,
                                                                               const int S1,
                                                                               float *__restrict__ out,
                                                                               const int outW,
                                                                               const int outH,
                                                                               const int pick ) {
    _cupy_correlate2D<float>( inp, inpW, inpH, kernel, kerW, kerH, S0, S1, out, outW, outH, pick );
}

extern "C" __global__ void __launch_bounds__(256 ) _cupy_correlate2D_float64( const double *__restrict__ inp,
                                                                               const int inpW,
                                                                               const int inpH,
                                                                               const double *__restrict__ kernel,
                                                                               const int kerW,
                                                                               const int kerH,
                                                                               const int S0,
                                                                               const int S1,
                                                                               double *__restrict__ out,
                                                                               const int outW,
                                                                               const int outH,
                                                                               const int pick ) {
    _cupy_correlate2D<double>( inp, inpW, inpH, kernel, kerW, kerH, S0, S1, out, outW, outH, pick );
}

extern "C" __global__ void __launch_bounds__(256 )
    _cupy_correlate2D_complex64( const thrust::complex<float> *__restrict__ inp,
                                 const int inpW,
                                 const int inpH,
                                 const thrust::complex<float> *__restrict__ kernel,
                                 const int kerW,
                                 const int kerH,
                                 const int S0,
                                 const int S1,
                                 thrust::complex<float> *__restrict__ out,
                                 const int outW,
                                 const int outH,
                                 const int pick ) {
    _cupy_correlate2D<thrust::complex<float>>( inp, inpW, inpH, kernel, kerW, kerH, S0, S1, out, outW, outH, pick );
}

extern "C" __global__ void __launch_bounds__( 256 )
    _cupy_correlate2D_complex128( const thrust::complex<double> *__restrict__ inp,
                                  const int inpW,
                                  const int inpH,
                                  const thrust::complex<double> *__restrict__ kernel,
                                  const int kerW,
                                  const int kerH,
                                  const int S0,
                                  const int S1,
                                  thrust::complex<double> *__restrict__ out,
                                  const int outW,
                                  const int outH,
                                  const int pick ) {
    _cupy_correlate2D<thrust::complex<double>>( inp, inpW, inpH, kernel, kerW, kerH, S0, S1, out, outW, outH, pick );
}


///////////////////////////////////////////////////////////////////////////////
//                              CONVOLVE 1D2O                                //
///////////////////////////////////////////////////////////////////////////////

template<typename T>
__device__ void _cupy_convolve1D2O( const T *__restrict__ inp,
                                const int inpW,
                                const T *__restrict__ kernel,
                                const int  kerW,
                                const int  kerH,
                                const int  mode,
                                T *__restrict__ out,
                                const int outW ) {

    const int tx { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
    const int stride { static_cast<int>( blockDim.x * gridDim.x ) };

    for ( int tid = tx; tid < outW; tid += stride ) {

        T temp {};

        if ( mode == 0 ) {  // Valid
            if ( tid >= 0 && tid < inpW ) {
                for ( int i = 0; i < kerW; i++ ) {
                    for ( int j = 0; j < kerH; j++ ) {
                        temp += inp[tid + kerW - i - 1] * inp[tid + kerH - j - 1] * kernel[ kerW * i + j];
                    }
                }
            }
        }
        out[tid] = temp;
    }

}

extern "C" __global__ void __launch_bounds__( 512 ) _cupy_convolve1D2O_int32( const int *__restrict__ inp,
                                                                          const int inpW,
                                                                          const int *__restrict__ kernel,
                                                                          const int  kerW,
                                                                          const int  kerH,
                                                                          const int  mode,
                                                                          int *__restrict__ out,
                                                                          const int outW ) {
    _cupy_convolve1D2O<int>( inp, inpW, kernel, kerW, kerH, mode, out, outW );
}

extern "C" __global__ void __launch_bounds__( 512 ) _cupy_convolve1D2O_int64( const long int *__restrict__ inp,
                                                                          const int inpW,
                                                                          const long int *__restrict__ kernel,
                                                                          const int  kerW,
                                                                          const int  kerH,
                                                                          const int  mode,
                                                                          long int *__restrict__ out,
                                                                          const int outW ) {
    _cupy_convolve1D2O<long int>( inp, inpW, kernel, kerW, kerH, mode, out, outW );
}

extern "C" __global__ void __launch_bounds__( 512 ) _cupy_convolve1D2O_float32( const float *__restrict__ inp,
                                                                            const int inpW,
                                                                            const float *__restrict__ kernel,
                                                                            const int  kerW,
                                                                            const int  kerH,
                                                                            const int  mode,
                                                                            float *__restrict__ out,
                                                                            const int outW ) {
    _cupy_convolve1D2O<float>( inp, inpW, kernel, kerW, kerH, mode, out, outW );
}

extern "C" __global__ void __launch_bounds__( 512 ) _cupy_convolve1D2O_float64( const double *__restrict__ inp,
                                                                            const int inpW,
                                                                            const double *__restrict__ kernel,
                                                                            const int  kerW,
                                                                            const int  kerH,
                                                                            const int  mode,
                                                                            double *__restrict__ out,
                                                                            const int outW ) {
    _cupy_convolve1D2O<double>( inp, inpW, kernel, kerW, kerH, mode, out, outW );
}

extern "C" __global__ void __launch_bounds__( 512 )
    _cupy_convolve1D2O_complex64( thrust::complex<float> *__restrict__ inp,
                              const int inpW,
                              thrust::complex<float> *__restrict__ kernel,
                              const int  kerW,
                              const int  kerH,
                              const int  mode,
                              thrust::complex<float> *__restrict__ out,
                              const int outW ) {
    _cupy_convolve1D2O<thrust::complex<float>>( inp, inpW, kernel, kerW, kerH, mode, out, outW );
}

extern "C" __global__ void __launch_bounds__( 512 )
    _cupy_convolve1D2O_complex128( const thrust::complex<double> *__restrict__ inp,
                               const int inpW,
                               const thrust::complex<double> *__restrict__ kernel,
                               const int  kerW,
                               const int  kerH,
                               const int  mode,
                               thrust::complex<double> *__restrict__ out,
                               const int outW ) {
    _cupy_convolve1D2O<thrust::complex<double>>( inp, inpW, kernel, kerW, kerH, mode, out, outW );
}



///////////////////////////////////////////////////////////////////////////////
//                              CONVOLVE 1D3O                                //
///////////////////////////////////////////////////////////////////////////////

template<typename T>
__device__ void _cupy_convolve1D3O( const T *__restrict__ inp,
                                const int inpW,
                                const T *__restrict__ kernel,
                                const int  kerW,
                                const int  kerH,
                                const int  kerD,
                                const int  mode,
                                T *__restrict__ out,
                                const int outW ) {

    const int tx { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
    const int stride { static_cast<int>( blockDim.x * gridDim.x ) };

    for ( int tid = tx; tid < outW; tid += stride ) {

        T temp {};

        if ( mode == 0 ) {  // Valid
            if ( tid >= 0 && tid < inpW ) {
                for ( int i = 0; i < kerW; i++ ) {
                    for ( int j = 0; j < kerH; j++ ) {
                        for ( int k = 0; k < kerD; k++ ) {
                            temp += inp[tid + kerW - i - 1] * inp[tid + kerH - j - 1] * inp[tid + kerD - k - 1] * kernel[ (kerW * i + j) * kerH + k ];
                        }
                    }
                }
            }
        }
        out[tid] = temp;
    }

}

extern "C" __global__ void __launch_bounds__( 512 ) _cupy_convolve1D3O_int32( const int *__restrict__ inp,
                                                                          const int inpW,
                                                                          const int *__restrict__ kernel,
                                                                          const int  kerW,
                                                                          const int  kerH,
                                                                          const int  kerD,
                                                                          const int  mode,
                                                                          int *__restrict__ out,
                                                                          const int outW ) {
    _cupy_convolve1D3O<int>( inp, inpW, kernel, kerW, kerH, kerD, mode, out, outW );
}

extern "C" __global__ void __launch_bounds__( 512 ) _cupy_convolve1D3O_int64( const long int *__restrict__ inp,
                                                                          const int inpW,
                                                                          const long int *__restrict__ kernel,
                                                                          const int  kerW,
                                                                          const int  kerH,
                                                                          const int  kerD,
                                                                          const int  mode,
                                                                          long int *__restrict__ out,
                                                                          const int outW ) {
    _cupy_convolve1D3O<long int>( inp, inpW, kernel, kerW, kerH, kerD, mode, out, outW );
}

extern "C" __global__ void __launch_bounds__( 512 ) _cupy_convolve1D3O_float32( const float *__restrict__ inp,
                                                                            const int inpW,
                                                                            const float *__restrict__ kernel,
                                                                            const int  kerW,
                                                                            const int  kerH,
                                                                            const int  kerD,
                                                                            const int  mode,
                                                                            float *__restrict__ out,
                                                                            const int outW ) {
    _cupy_convolve1D3O<float>( inp, inpW, kernel, kerW, kerH, kerD, mode, out, outW );
}

extern "C" __global__ void __launch_bounds__( 512 ) _cupy_convolve1D3O_float64( const double *__restrict__ inp,
                                                                            const int inpW,
                                                                            const double *__restrict__ kernel,
                                                                            const int  kerW,
                                                                            const int  kerH,
                                                                            const int  kerD,
                                                                            const int  mode,
                                                                            double *__restrict__ out,
                                                                            const int outW ) {
    _cupy_convolve1D3O<double>( inp, inpW, kernel, kerW, kerH, kerD, mode, out, outW );
}

extern "C" __global__ void __launch_bounds__( 512 )
    _cupy_convolve1D3O_complex64( thrust::complex<float> *__restrict__ inp,
                              const int inpW,
                              thrust::complex<float> *__restrict__ kernel,
                              const int  kerW,
                              const int  kerH,
                              const int  kerD,
                              const int  mode,
                              thrust::complex<float> *__restrict__ out,
                              const int outW ) {
    _cupy_convolve1D3O<thrust::complex<float>>( inp, inpW, kernel, kerW, kerH, kerD, mode, out, outW );
}

extern "C" __global__ void __launch_bounds__( 512 )
    _cupy_convolve1D3O_complex128( const thrust::complex<double> *__restrict__ inp,
                               const int inpW,
                               const thrust::complex<double> *__restrict__ kernel,
                               const int  kerW,
                               const int  kerH,
                               const int  kerD,
                               const int  mode,
                               thrust::complex<double> *__restrict__ out,
                               const int outW ) {
    _cupy_convolve1D3O<thrust::complex<double>>( inp, inpW, kernel, kerW, kerH, kerD, mode, out, outW );
}
