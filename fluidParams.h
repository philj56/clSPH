#ifndef FLUID_PARAMS_H
#define FLUID_PARAMS_H

#ifdef OPENCL_COMPILING
	#define OPENCL_FLOAT3 float3
	#define OPENCL_FLOAT float
#else
	#include <math.h>
	#ifdef MAC
		#include <OpenCL/cl.h>
	#else
		#include <CL/cl.h>
	#endif

	#define OPENCL_FLOAT3 cl_float3
	#define OPENCL_FLOAT cl_float
#endif /* OPENCL_COMPILING */

#define MAX_NEIGHBOURS 400


struct fluidParams {
	OPENCL_FLOAT3 gravity;	
	OPENCL_FLOAT timeStep;
	OPENCL_FLOAT relaxationFactor;
};
#endif /* FLUID_PARAMS_H */
