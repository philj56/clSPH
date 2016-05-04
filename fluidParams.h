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
	OPENCL_FLOAT3 gravity;	
	OPENCL_FLOAT timeStep;

	OPENCL_FLOAT relaxationFactor;
};

#ifndef OPENCL_COMPILING

	/* Spline weight from Monaghan 1992, first derivative w.r.t. r */
	OPENCL_FLOAT weightMonaghanSplinePrime(OPENCL_FLOAT r, OPENCL_FLOAT interactionRadius)
	{
		OPENCL_FLOAT w = 0.0f;
	
		if (r < interactionRadius)
		{
			w += r * (2.25f * r - 3.0f * interactionRadius);
		}
		else if (r < 2.0f * interactionRadius)
		{
			w += -3.0f * pow(interactionRadius - 0.5f * r, 2);
		}

		w *= pow(interactionRadius, -3);
		
		return w;
	}

	OPENCL_FLOAT weightMonaghanSpline(OPENCL_FLOAT r, OPENCL_FLOAT interactionRadius)
	{
		OPENCL_FLOAT q;
		OPENCL_FLOAT w;
	
		q = r / interactionRadius;
		w = 0.0f;
	
		if (islessequal(q, 1.0f))
		{
			w += 1.0f + q * q * (q * 0.75f - 1.5f);
		}
		else if (islessequal(q, 2.0f))
		{
			w += 0.25f * pow(2.0f - q, 3.0f);
		}
	
		return w;
	}
	
	/* Deduce attributes from timeStep, restDensity, particleRadius & viscosity */
	/* Use spline weight from Monaghan 1992 */
	void updateFluidParams (struct fluidParams *params)
	{
		params->monaghanSplineNormalisation = 1.0f / (M_PI * pow(params->interactionRadius, 3));
		params->monaghanSplinePrimeNormalisation = 10.0f / (7.0f * pow(params->interactionRadius, 3));
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
