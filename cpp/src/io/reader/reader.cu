/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstdio>
#include <memory>
#include <string>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cuComplex.h>

#ifdef USE_NVTX
#include <nvtx3/nvToolsExt.h>

const uint32_t colors[]   = { 0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff };
const int      num_colors = sizeof( colors ) / sizeof( uint32_t );

#define PUSH_RANGE( name, cid )                                                                                        \
    {                                                                                                                  \
        int color_id                      = cid;                                                                       \
        color_id                          = color_id % num_colors;                                                     \
        nvtxEventAttributes_t eventAttrib = { 0 };                                                                     \
        eventAttrib.version               = NVTX_VERSION;                                                              \
        eventAttrib.size                  = NVTX_EVENT_ATTRIB_STRUCT_SIZE;                                             \
        eventAttrib.colorType             = NVTX_COLOR_ARGB;                                                           \
        eventAttrib.color                 = colors[color_id];                                                          \
        eventAttrib.messageType           = NVTX_MESSAGE_TYPE_ASCII;                                                   \
        eventAttrib.message.ascii         = name;                                                                      \
        nvtxRangePushEx( &eventAttrib );                                                                               \
    }
#define POP_RANGE nvtxRangePop( );
#else
#define PUSH_RANGE( name, cid )
#define POP_RANGE
#endif

#define CUDA_RT_CALL( call )                                                                                           \
    {                                                                                                                  \
        cudaError_t cudaStatus = call;                                                                                 \
        if ( cudaSuccess != cudaStatus )                                                                               \
            fprintf( stderr,                                                                                           \
                     "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "                                        \
                     "with "                                                                                           \
                     "%s (%d).\n",                                                                                     \
                     #call,                                                                                            \
                     __LINE__,                                                                                         \
                     __FILE__,                                                                                         \
                     cudaGetErrorString( cudaStatus ),                                                                 \
                     cudaStatus );                                                                                     \
    }

__global__ void
cuda_parser( const int data_size, const cuFloatComplex *__restrict__ input, cuFloatComplex *__restrict__ output ) {

    unsigned int tx { blockIdx.x * blockDim.x + threadIdx.x };
    unsigned int stride { blockDim.x * gridDim.x };

    for ( unsigned int tid = tx; tid < data_size; tid += stride ) {

        int stop  = 40012800;
        int start = stop - 10;
        if ( ( tid > start ) && ( tid < stop ) ) {
            printf( "%d: %f, %f\n", tid, input[tid].x, input[tid].y );
        }
        // char temp[8] { };

        // for ( int i = 0; i < 8; i++ ) {
        //     temp[i] = input[tid * 8 + i];
        // }
    }
}

int main( int argc, char **argv ) {

    // PUSH_RANGE( "warm-up", 0 )
    // char *d_b { };
    // CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &d_b ), 8 ) );
    // POP_RANGE

    std::string filename { };

    if ( argc > 1 ) {
        printf( "%s\n", argv[1] );
        filename = argv[1];
    } else {
        throw std::runtime_error( "Error: Must pass binary file\n" );
    }

    //-----------------------------------------------------------------------------
    // memory map in the data
    PUSH_RANGE( "mmap", 0 )
    char *map_data { nullptr };
    // cuFloatComplex *h_parsed { nullptr };
    struct stat st {};
    int         fd { };

    if ( ( fd = open( filename.c_str( ), O_RDONLY ) ) == -1 ) {
        throw std::runtime_error( "Error: Opening file\n" );
    }

    if ( ( fstat( fd, &st ) ) == -1 ) {
        throw std::runtime_error( "Error: Can't stat file\n" );
    }

    size_t num_bytes = st.st_size;

    printf( "num_bytes = %lu\n", num_bytes );

    map_data = reinterpret_cast<char *>( mmap( 0, num_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0 ) );

    if ( map_data == MAP_FAILED || num_bytes == 0 ) {
        throw std::runtime_error( "Error: Mapping file\n" );
    }
    POP_RANGE

    //-----------------------------------------------------------------------------
    // pinned mapped memory
    PUSH_RANGE( "pin memory", 1 )
    // CUDA_RT_CALL( cudaHostRegister( map_data, num_bytes, cudaHostRegisterDefault ) );
    // CUDA_RT_CALL( cudaHostAlloc( &h_parsed, num_bytes, cudaHostRegisterDefault ) );
    POP_RANGE

    //-----------------------------------------------------------------------------
    // setup gpu
    PUSH_RANGE( "setup gpu", 2 )
    // cudaStream_t stream;
    // CUDA_RT_CALL( cudaStreamCreate( &stream ) );

    cuFloatComplex *d_binary { };
    // void *          d_parsed { };

    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &d_binary ), num_bytes ) );
    // CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &d_parsed ), num_bytes ) );

    CUDA_RT_CALL( cudaMemcpy( d_binary, map_data, num_bytes, cudaMemcpyHostToDevice ) );

    // CUDA_RT_CALL( cudaStreamSynchronize( stream ) );
    POP_RANGE

    //-----------------------------------------------------------------------------
    // gpu kernel
    // PUSH_RANGE( "gpu kernel", 3 )
    // dim3 threadPerBlock { 1024 };
    // dim3 blocksPerGrid { 1024 };

    // int data_size = num_bytes / 8;
    // printf( "data_size = %d\n", data_size );

    // void *args[] { &data_size, &d_binary, &d_parsed };

    // CUDA_RT_CALL(
    //     cudaLaunchKernel( reinterpret_cast<void *>( &cuda_parser ), blocksPerGrid, threadPerBlock, args, 0, stream ) );

    // // CUDA_RT_CALL( cudaMemcpyAsync( h_parsed, d_parsed, num_bytes, cudaMemcpyDeviceToHost, stream ) );

    // CUDA_RT_CALL( cudaStreamSynchronize( stream ) );
    // POP_RANGE

    //-----------------------------------------------------------------------------
    // clean up
    PUSH_RANGE( "clean up", 4 )
    // CUDA_RT_CALL( cudaStreamDestroy( stream ) );

    CUDA_RT_CALL( cudaFree( d_binary ) );
    // CUDA_RT_CALL( cudaFree( d_parsed ) );
    POP_RANGE

    return ( EXIT_SUCCESS );
}