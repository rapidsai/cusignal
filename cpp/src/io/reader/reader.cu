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

#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace std::chrono_literals;

#define CUDA_RT_CALL(call)                                                                  \
    {                                                                                       \
        cudaError_t cudaStatus = call;                                                      \
        if (cudaSuccess != cudaStatus)                                                      \
            fprintf(stderr,                                                                 \
                    "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "              \
                    "with "                                                                 \
                    "%s (%d).\n",                                                           \
                    #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus); \
    }

int main( int argc, char **argv ) {

    std::string filename { };

    if ( argc > 1 ) {
        printf( "%s\n", argv[1] );
        filename = argv[1];
    } else {
        throw std::runtime_error( "Error: Must pass binary file\n" );
    }

    //-----------------------------------------------------------------------------
    // memory map in the data
    char *      map_data { nullptr };
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

    map_data = reinterpret_cast<char *>(mmap( 0, num_bytes, PROT_READ|PROT_WRITE, MAP_PRIVATE, fd, 0 ));

    if ( map_data == MAP_FAILED || num_bytes == 0 ) {
		throw std::runtime_error( "Error: Mapping file\n" );
	}

	//-----------------------------------------------------------------------------
	// pinned mapped memory
	CUDA_RT_CALL(cudaHostRegister(map_data, num_bytes, cudaHostRegisterDefault));

	//-----------------------------------------------------------------------------
	// setup gpu
	cudaStream_t stream;
	CUDA_RT_CALL(cudaStreamCreate(&stream));
	

	char * d_a {};

	CUDA_RT_CALL(cudaMalloc(reinterpret_cast<void **>(&d_a), num_bytes));
	CUDA_RT_CALL(cudaMemcpyAsync(d_a, map_data, num_bytes, cudaMemcpyHostToDevice, stream));

	std::this_thread::sleep_for(5s);

	//-----------------------------------------------------------------------------
	// clean up
	CUDA_RT_CALL(cudaStreamDestroy(stream));

	CUDA_RT_CALL(cudaFree(d_a));

    return ( EXIT_SUCCESS );
}