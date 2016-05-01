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

	OPENCL_FLOAT surface_tension_coefficient;
	OPENCL_FLOAT surface_tension_term;
	OPENCL_FLOAT surface_tension_normalization;

	OPENCL_FLOAT viscosity;
	OPENCL_FLOAT3 gravity;	
	OPENCL_FLOAT timeStep;

	OPENCL_FLOAT pressureScalingFactor;
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
	
	/* Deduce attributes from timeStep, restDensity, particleRadius & viscosity */
	/* Use spline weight from Monaghan 1992 */
	void updateFluidParams (struct fluidParams *params)
	{
		params->monaghanSplineNormalisation = 1.0f / (M_PI * pow(params->interactionRadius, 3));
		params->monaghanSplinePrimeNormalisation = 10.0f / (7.0f * pow(params->interactionRadius, 3));

		const OPENCL_FLOAT beta = 2.0f * pow (params->timeStep * params->particleMass / params->restDensity, 2);

		/* Sum of (kernel gradients) */
		OPENCL_FLOAT delKernelSum[3] = {0.0f, 0.0f, 0.0f};
		
		/* Sum of (kernel gradients) magnitude squared */
		OPENCL_FLOAT delKernelSumDotDelKernelSum = 0.0f;
		
		/* Sum of (kernel gradients magnitude squared) */
		OPENCL_FLOAT delKernelDotDelKernelSum = 0.0f;

		for (OPENCL_FLOAT x = -2.0f * params->interactionRadius; x <= 2.0f * params->interactionRadius; x += 2.0f * params->particleRadius)
		{
			for (OPENCL_FLOAT y = -2.0f * params->interactionRadius; y <= 2.0f * params->interactionRadius; y += 2.0f * params->particleRadius)
			{
				for (OPENCL_FLOAT z = -2.0f * params->interactionRadius; z <= 2.0f * params->interactionRadius; z += 2.0f * params->particleRadius)
				{
					OPENCL_FLOAT r2 = x * x + y * y + z * z;
					OPENCL_FLOAT r = sqrt(r2);

					if(r < params->interactionRadius && r != 0.0f)
					{
						OPENCL_FLOAT factor = params->monaghanSplinePrimeNormalisation * weightMonaghanSplinePrime(r, params->interactionRadius);
						factor /= r;

						OPENCL_FLOAT delKernel[3] = {
							factor * x,
							factor * y,
							factor * z
						};

						for (size_t i = 0; i < 3; i++)
						{
							/* Sum kernel gradients */
							delKernelSum[i] += delKernel[i];

							/* Sum (kernel gradients magnitude squared) */
							delKernelDotDelKernelSum += pow (delKernel[i], 2);
						}
							
					}
				}
			}
		}

		for (size_t i = 0; i < 3; i++)
		{
			delKernelSumDotDelKernelSum += pow (delKernelSum[i], 2);
		}
//		printf("Mass: %f\nIntRad: %f\nNormalisation: %f\nBeta: %f\nDelDotSum: %f\nDotDelSum: %f\n", params->particleMass, params->interactionRadius, params->monaghanSplineNormalisation, beta, delKernelDotDelKernelSum, delKernelSumDotDelKernelSum);

		params->pressureScalingFactor = -1.0f / (beta * (-delKernelSumDotDelKernelSum - delKernelDotDelKernelSum));
		printf("scale: %f\n", params->pressureScalingFactor);
//		params->pressureScalingFactor /= 50.0f;
	}

	/* Create fully initialised fluidParams from key inputs */
	struct fluidParams newFluidParams (OPENCL_FLOAT particleMass, 
			 		   OPENCL_FLOAT particleRadius, 
					   OPENCL_FLOAT restDensity,
					   OPENCL_FLOAT interactionRadius,
					   OPENCL_FLOAT timeStep, 
					   OPENCL_FLOAT viscosity, 
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
			.gravity           = gravity
		};
	
		updateFluidParams (&params);
	
		return params;
	}

#endif /* OPENCL_COMPILING */

#endif /* FLUID_PARAMS_H */
