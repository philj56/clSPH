#ifndef HASH_H
#define HASH_H

#include "cl_types.h"

#define MAX_HASH 18

struct hash {
	OPENCL_UINT indices[MAX_HASH]; 
	OPENCL_UCHAR num_indices;
};

OPENCL_UINT hash(OPENCL_FLOAT3 pos, OPENCL_FLOAT h, OPENCL_UINT size)
{
	return (((OPENCL_UINT)(pos.x / h) * 73856093) ^ ((OPENCL_UINT)(pos.y / h) * 19349663) ^ ((OPENCL_UINT)(pos.z / h) * 83492791)) % size;
}

#endif /* HASH_H */
