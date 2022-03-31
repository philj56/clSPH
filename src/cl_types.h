#ifndef CLSPH_CL_TYPES_H
#define CLSPH_CL_TYPES_H

#ifdef OPENCL_COMPILING
	#define OPENCL_UCHAR uchar
	#define OPENCL_FLOAT3 float3
	#define OPENCL_FLOAT float
	#define OPENCL_UINT uint
	#define OPENCL_INT int
#else
	#include <stdbool.h>
	#ifdef __APPLE__
		#include <OpenCL/cl.h>
	#else
		#include <CL/cl.h>
	#endif

	#define OPENCL_FLOAT3 cl_float3
	#define OPENCL_FLOAT cl_float
	#define OPENCL_UCHAR cl_uchar
	#define OPENCL_UINT cl_uint
	#define OPENCL_INT cl_int
#endif /* OPENCL_COMPILING */

#endif /* CLSPH_CL_TYPES_H */
