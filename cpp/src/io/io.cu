


#include <cupy/complex.cuh>

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
    #elif FLAG == 6 || FLAG == 8
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

    #elif FLAG == 7 || FLAG == 9
    __device__ double swap_float( double val )
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

    #endif

    __global__ void _cupy_unpack(
        const size_t N,
        const bool little,
        unsigned char * __restrict__ input,
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
                    ${datatype} temp = swap_float(data);
                    output[tid] = temp;
                #elif FLAG == 8
                    float real = swap_float(data.real());
                    float imag = swap_float(data.imag());

                    output[tid] = complex<float>(real, imag);
                #elif FLAG == 9
                    double real = swap_float(data.real());
                    double imag = swap_float(data.imag());

                    output[tid] = complex<double>(real, imag);
                #endif
            }
        }
    }

__global__ void _cupy_pack(
	const size_t N,
	${datatype} * __restrict__ input,
	unsigned char * __restrict__ output) {

	 const int tx {
		static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) };
	const int stride { static_cast<int>(blockDim.x * gridDim.x) };

	for ( int tid = tx; tid < N; tid += stride ) {
		output[tid] = reinterpret_cast<unsigned char*>(input)[tid];
	}
}


