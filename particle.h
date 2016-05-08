#ifndef PARTICLE_H
#define PARTICLE_H

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

struct particle {
	OPENCL_FLOAT3 pos;
	OPENCL_FLOAT3 vel;
	OPENCL_FLOAT  density;
	OPENCL_FLOAT  pressure;
	OPENCL_FLOAT  advection;
	OPENCL_FLOAT  densityAdvection;
	OPENCL_FLOAT3 velocityAdvection;
	OPENCL_FLOAT3 displacement;
	OPENCL_FLOAT3 sumPressureMovement;
	OPENCL_FLOAT3 normal;
};

#ifndef OPENCL_COMPILING
	const struct particle defaultParticle =
	{
		{{0.0f}},
		{{0.0f}},
		0.0f,
		0.0f,	
		0.0f,
		0.0f,
		{{0.0f}},
		{{0.0f}},
		{{0.0f}},
		{{0.0f}}	
	};
#endif

#endif /* PARTICLE_H */
