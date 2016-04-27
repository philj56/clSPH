#ifndef FLUID_PARAMS_H
#define FLUID_PARAMS_H

#ifdef OPENCL_COMPILING
	#define OPENCL_FLOAT3 float3
	#define OPENCL_FLOAT float
#else
	#ifdef MAC
		#include <OpenCL/cl.h>
	#else
		#include <CL/cl.h>
	#endif

	#define OPENCL_FLOAT3 cl_float3
	#define OPENCL_FLOAT cl_float
#endif /* OPENCL_COMPILING */

struct fluidParams {
	OPENCL_FLOAT particleMass;
	OPENCL_FLOAT particleRadius;
	OPENCL_FLOAT restDensity;	
	OPENCL_FLOAT interactionRadius;
	
	OPENCL_FLOAT surface_tension_coefficient;
	OPENCL_FLOAT surface_tension_term;
	OPENCL_FLOAT surface_tension_normalization;

	OPENCL_FLOAT viscosity;
	OPENCL_FLOAT3 gravity;	
	OPENCL_FLOAT timeStep;

	OPENCL_FLOAT pressureScalingFactor;
};

#endif
