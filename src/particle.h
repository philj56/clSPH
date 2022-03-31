#ifndef CLSPH_PARTICLE_H
#define CLSPH_PARTICLE_H

#include "cl_types.h"

struct particle {
	OPENCL_FLOAT3 pos;
	OPENCL_FLOAT3 vel;

	OPENCL_FLOAT3 velocity_advection;
	OPENCL_FLOAT3 displacement;
	OPENCL_FLOAT3 sum_pressure_movement;
	OPENCL_FLOAT3 normal;
	
	OPENCL_FLOAT  density;
	OPENCL_FLOAT  pressure;
	OPENCL_FLOAT  advection;
	OPENCL_FLOAT  density_advection;
	OPENCL_FLOAT  kernel_correction;	

	OPENCL_FLOAT  volume;

	OPENCL_UINT   fluid_index;
};

#endif /* CLSPH_PARTICLE_H */
