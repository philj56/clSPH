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


typedef struct {
	OPENCL_FLOAT3 gravity;	
	OPENCL_FLOAT  timeStep;
	OPENCL_FLOAT  relaxationFactor;
	OPENCL_FLOAT  gridSize;
	OPENCL_FLOAT  mass;
	OPENCL_FLOAT  radius;
	OPENCL_FLOAT  restDensity;	
	OPENCL_FLOAT  interactionRadius;

	OPENCL_FLOAT  monaghanSplineNormalisation;
	OPENCL_FLOAT  monaghanSplinePrimeNormalisation;

	OPENCL_FLOAT  surfaceTension;
	OPENCL_FLOAT  surfaceTensionTerm;
	OPENCL_FLOAT  surfaceTensionNormalisation;
	
	OPENCL_FLOAT  adhesion;
	OPENCL_FLOAT  adhesionNormalisation;
	
	OPENCL_FLOAT  viscosity;
} fluid_t;
	
#ifndef OPENCL_COMPILING
	void updateDeducedParams (fluid_t *fluid)
	{
		fluid->monaghanSplineNormalisation = 1.0f / (M_PI * pow(fluid->interactionRadius * 0.5f, 3));
		fluid->monaghanSplinePrimeNormalisation = 10.0f / (7.0f * pow(fluid->interactionRadius * 0.5f, 3));
		fluid->surfaceTensionNormalisation = 32.0f / (M_PI * pow(fluid->interactionRadius, 9));
		fluid->surfaceTensionTerm = -pow(fluid->interactionRadius, 6) / 64.0f;
		fluid->adhesionNormalisation = 0.007f * pow(fluid->interactionRadius, -3.25);
	}
#endif /* OPENCL_COMPILING */

#endif /* FLUID_PARAMS_H */
