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
	OPENCL_FLOAT3 velocityAdvection;
	OPENCL_FLOAT3 displacement;
	OPENCL_FLOAT3 sumPressureMovement;
	OPENCL_FLOAT3 normal;
	
	OPENCL_FLOAT  density;
	OPENCL_FLOAT  pressure;
	OPENCL_FLOAT  advection;
	OPENCL_FLOAT  densityAdvection;
	OPENCL_FLOAT  kernelCorrection;
	
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
};

struct boundaryParticle {
	OPENCL_FLOAT3 pos;
	OPENCL_FLOAT  interactionRadius;
	OPENCL_FLOAT  volume;
	OPENCL_FLOAT  adhesionModifier;
	OPENCL_FLOAT  viscosity;
	OPENCL_FLOAT  monaghanSplineNormalisation;
};

#ifndef OPENCL_COMPILING
	/* Default particle - 0.1m^3 water particle at origin */
	const struct particle defaultParticle =
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
		1.0f,
		0.1f,
		1000.0f,
		0.225f,
		223.56f,
		1003.33f,
		0.0725f,
		-0.000002027f,
		6892193.195f,
		0.0725f,
		0.892288f,
		0.00089f
	};

	const struct boundaryParticle defaultBoundaryParticle =
	{
		{{0.0f}},
		0.225f,
		0.0f,
		1.0f,
		0.0f,
		223.56f
	};

	void updateDeducedParams (struct particle *particle)
	{
		particle->monaghanSplineNormalisation = 1.0f / (M_PI * pow(particle->interactionRadius * 0.5f, 3));
		particle->monaghanSplinePrimeNormalisation = 10.0f / (7.0f * pow(particle->interactionRadius * 0.5f, 3));
		particle->surfaceTensionNormalisation = 32.0f / (M_PI * pow(particle->interactionRadius, 9));
		particle->surfaceTensionTerm = -pow(particle->interactionRadius, 6) / 64.0f;
		particle->adhesionNormalisation = 0.007f * pow(particle->interactionRadius, -3.25);
	}
	
	void updateDeducedBoundaryParams (struct boundaryParticle *particle)
	{
		particle->monaghanSplineNormalisation = 1.0f / (M_PI * pow(particle->interactionRadius * 0.5f, 3));
	}

#endif

#endif /* PARTICLE_H */
