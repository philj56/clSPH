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
	OPENCL_FLOAT particleMass;
	OPENCL_FLOAT particleRadius;
	OPENCL_FLOAT restDensity;	
	OPENCL_FLOAT interactionRadius;

	OPENCL_FLOAT monaghanSplineNormalisation;
	OPENCL_FLOAT monaghanSplinePrimeNormalisation;

	OPENCL_FLOAT surfaceTension;
	OPENCL_FLOAT surfaceTensionTerm;
	OPENCL_FLOAT surfaceTensionNormalisation;
	
	OPENCL_FLOAT viscosity;
	OPENCL_FLOAT timeStep;
	OPENCL_FLOAT relaxationFactor;

	OPENCL_FLOAT3 gravity;	
};

#ifndef OPENCL_COMPILING
	/* Deduce attributes from timeStep, restDensity, particleRadius & viscosity */
	/* Use spline weight from Monaghan 1992 */
	void updateFluidParams (struct fluidParams *params)
	{
		params->monaghanSplineNormalisation = 1.0f / (M_PI * pow(params->interactionRadius * 0.5f, 3));
		params->monaghanSplinePrimeNormalisation = 10.0f / (7.0f * pow(params->interactionRadius * 0.5f, 3));
		params->surfaceTensionNormalisation = 32.0f / (M_PI * pow(params->interactionRadius, 9));
		params->surfaceTensionTerm = -pow(params->interactionRadius, 6) / 64.0f;
		params->relaxationFactor = 0.5f;
	}

	/* Create fully initialised fluidParams from key inputs */
	struct fluidParams newFluidParams (OPENCL_FLOAT particleMass, 
			 		   OPENCL_FLOAT particleRadius, 
					   OPENCL_FLOAT restDensity,
					   OPENCL_FLOAT interactionRadius,
					   OPENCL_FLOAT timeStep, 
					   OPENCL_FLOAT viscosity, 
					   OPENCL_FLOAT surfaceTension, 
					   OPENCL_FLOAT3 gravity)
	{
		struct fluidParams params =
		{	
			.particleMass      = particleMass,
			.particleRadius    = particleRadius,
			.restDensity       = restDensity,
			.interactionRadius = interactionRadius,
			.timeStep          = timeStep,
			.viscosity         = viscosity,
			.surfaceTension    = surfaceTension,
			.gravity           = gravity
		};	

		updateFluidParams (&params);
	
		return params;
	}

#endif /* OPENCL_COMPILING */

#endif /* FLUID_PARAMS_H */
