#ifndef CLSPH_FLUID_PARAMS_H
#define CLSPH_FLUID_PARAMS_H

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


struct fluid_params {
	OPENCL_FLOAT3 gravity;	
	OPENCL_FLOAT  time_step;
	OPENCL_FLOAT  relaxation_factor;
	OPENCL_FLOAT  grid_size;
	OPENCL_FLOAT  mass;
	OPENCL_FLOAT  radius;
	OPENCL_FLOAT  rest_density;	
	OPENCL_FLOAT  interaction_radius;

	OPENCL_FLOAT  monaghan_spline_normalisation;
	OPENCL_FLOAT  monaghan_spline_prime_normalisation;

	OPENCL_FLOAT  surface_tension;
	OPENCL_FLOAT  surface_tension_term;
	OPENCL_FLOAT  surface_tension_normalisation;
	
	OPENCL_FLOAT  adhesion;
	OPENCL_FLOAT  adhesion_normalisation;
	
	OPENCL_FLOAT  viscosity;
};
	
#ifndef OPENCL_COMPILING
void update_deduced_params (struct fluid_params *fluid);
#endif /* OPENCL_COMPILING */

#endif /* CLSPH_FLUID_PARAMS_H */
