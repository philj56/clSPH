#ifndef PARTICLE_H
#define PARTICLE_H

#ifdef OPENCL_COMPILING
	#define OPENCL_FLOAT3 float3
	#define OPENCL_FLOAT float
	#define OPENCL_UINT uint
#else
	#include <math.h>
	#ifdef MAC
		#include <OpenCL/cl.h>
	#else
		#include <CL/cl.h>
	#endif

	#define OPENCL_FLOAT3 cl_float3
	#define OPENCL_FLOAT cl_float
	#define OPENCL_UINT cl_uint
#endif /* OPENCL_COMPILING */

typedef struct {
	OPENCL_FLOAT3 pos;
	OPENCL_FLOAT3 vel;
	OPENCL_FLOAT3 velocityAdvection;
	OPENCL_FLOAT3 displacement;
	OPENCL_FLOAT3 sumPressureMovement;
	OPENCL_FLOAT3 normal;
	
	OPENCL_FLOAT  density;
	OPENCL_FLOAT  pressure;
	OPENCL_FLOAT  advection;
	OPENCL_FLOAT  densityAdvection;
	OPENCL_FLOAT  kernelCorrection;	

	OPENCL_FLOAT  volume;

	OPENCL_UINT   fluidIndex;

	bool is_static_boundary;
} particle_t;

#ifndef OPENCL_COMPILING
	/* Default particle - 0.1m^3 water particle at origin */
	const particle_t defaultParticle =
	{
		{{0.0f}},
		{{0.0f}},
		{{0.0f}},
		{{0.0f}},
		{{0.0f}},
		{{0.0f}},	
		0.0f,
		0.0f,	
		0.0f,
		0.0f,
		1.0f,
		0.0f,
		0,
		true
	};	
#endif /* OPENCL_COMPILING */

#endif /* PARTICLE_H */
