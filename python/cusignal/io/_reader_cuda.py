# Copyright (c) 2019-2020, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cupy as cp

from string import Template

from ..utils._caches import _cupy_kernel_cache


# Custom Cupy raw kernel implementing binary readers
# Matthew Nicely - mnicely@nvidia.com
_cupy_unpack_sigmf_src = Template(
    """
${header}

#define FLAG ${flag}

/*
    0 = int8
    1 = uint8
    2 = int16
    3 = uint16
    4 = int32
    5 = uint32
    6 = float32
    7 = complex64
*/

extern "C" {

    #if FLAG == 2
    // Byte swap short
    __device__ short swap_int16( short val )
    {
        return (val << 8) | ((val >> 8) & 0xFF);
    }

    #elif FLAG == 3
    // Byte swap unsigned short
    __device__ unsigned short swap_uint16( unsigned short val )
    {
        return (val << 8) | (val >> 8 );
    }

    #elif FLAG == 4
    // Byte swap int
    __device__ int swap_int32( int val )
    {
        val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF );
        return (val << 16) | ((val >> 16) & 0xFFFF);
    }

    #elif FLAG == 5
    // Byte swap unsigned int
    __device__ unsigned int swap_uint32( unsigned int val )
    {
        val = ((val << 8) & 0xFF00FF00 ) | ((val >> 8) & 0xFF00FF );
        return (val << 16) | (val >> 16);
    }

    // Byte swap float
    #elif FLAG > 5
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
    #endif

    __global__ void _cupy_unpack_sigmf(
        const size_t N,
        const int data_size,
        const bool little,
        unsigned int * __restrict__ input,
        ${datatype} * __restrict__ output) {

         const int tx {
            static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) };
        const int stride { static_cast<int>(blockDim.x * gridDim.x) };

        for ( int tid = tx; tid < N; tid += stride ) {

            if ( little ) {
                output[tid] = reinterpret_cast<${datatype}*>(input)[tid];
            } else {
                ${datatype} data = reinterpret_cast<${datatype}*>(input)[tid];

                #if FLAG < 2
                    output[tid] = data;
                #elif FLAG == 2
                    ${datatype} temp = swap_int16(data);
                    output[tid] = temp;
                #elif FLAG == 3
                    ${datatype} temp = swap_uint16(data);
                    output[tid] = temp;
                #elif FLAG == 4
                    ${datatype} temp = swap_int32(data);
                    output[tid] = temp;
                #elif FLAG == 5
                    ${datatype} temp = swap_uint32(data);
                    output[tid] = temp;
                #elif FLAG == 6
                    ${datatype} temp = swap_float(data);
                    output[tid] = temp;
                #elif FLAG == 7
                    float real = swap_float(data.real());
                    float imag = swap_float(data.imag());

                    output[tid] = complex<float>(real, imag);
                #endif
            }
        }
    }
}
"""
)


class _cupy_reader_wrapper(object):
    def __init__(self, grid, block, kernel):
        if isinstance(grid, int):
            grid = (grid,)
        if isinstance(block, int):
            block = (block,)

        self.grid = grid
        self.block = block
        self.kernel = kernel

    def __call__(self, out_size, data_size, little, binary, out):

        kernel_args = (out_size, data_size, little, binary, out)

        self.kernel(self.grid, self.block, kernel_args)


def _get_backend_kernel(
    dtype, grid, block, k_type,
):
    from ..utils.compile_kernels import GPUKernel

    kernel = _cupy_kernel_cache[(str(dtype), k_type.value)]
    if kernel:
        if k_type == GPUKernel.UNPACK_SIGMF:
            return _cupy_reader_wrapper(grid, block, kernel)

        raise ValueError(
            "Kernel {} not found in _cupy_kernel_cache".format(k_type)
        )


def _unpack(binary, dtype, endianness):

    from ..utils.compile_kernels import _populate_kernel_cache, GPUKernel

    data_size = cp.dtype(dtype).itemsize // binary.dtype.itemsize

    out_size = binary.shape[0] // data_size

    out = cp.empty_like(binary, dtype=dtype, shape=out_size)

    if endianness == "B":
        little = False
    else:
        little = True

    device_id = cp.cuda.Device()
    numSM = device_id.attributes["MultiProcessorCount"]
    blockspergrid = numSM * 20
    threadsperblock = 512

    _populate_kernel_cache(out.dtype, GPUKernel.UNPACK_SIGMF)
    kernel = _get_backend_kernel(
        out.dtype, blockspergrid, threadsperblock, GPUKernel.UNPACK_SIGMF,
    )

    kernel(out_size, data_size, little, binary, out)

    # Remove binary data
    del binary

    return out
