#ifndef HASH_H
#define HASH_H

#ifdef OPENCL_COMPILING
	#define OPENCL_UCHAR uchar
	#define OPENCL_UINT uint
	#define OPENCL_FLOAT3 float3
	#define OPENCL_FLOAT float
#else
	#ifdef MAC
		#include <OpenCL/cl.h>
	#else
		#include <CL/cl.h>
	#endif

	#define OPENCL_UCHAR cl_uchar
	#define OPENCL_UINT cl_uint
	#define OPENCL_FLOAT3 cl_float3
	#define OPENCL_FLOAT cl_float
#endif /* OPENCL_COMPILING */

#define MAX_HASH 18

typedef struct {
	OPENCL_UINT indices[MAX_HASH]; 
	OPENCL_UCHAR num_indices;
} hash_t;

OPENCL_UINT hash(OPENCL_FLOAT3 pos, OPENCL_FLOAT h, OPENCL_UINT size)
{
	return ((OPENCL_UINT)((pos.x / h) * 73856093) ^ (OPENCL_UINT)((pos.y / h) * 19349663) ^ (OPENCL_UINT)((pos.z / h) * 83492791)) % size;
}

#endif /* HASH_H */
