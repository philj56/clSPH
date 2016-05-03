#define OPENCL_COMPILING
#include "fluidParams.h"

#define MAX_NEIGHBOURS 200

float weightMonaghanSpline(float3 p12, float interactionRadius);

float weightMonaghanSplinePrime(float3 p12, float interactionRadius);

float weightSurfaceTension(float3 p12, float h, float extraTerm);

float viscosityMonaghan(float3 p12,
						float3 v12,
						float  rhoAvg,
						float  interactionRadius);

/* Simple neighbour search. TODO: optimise so neighbours aren't repeatedly found */
__kernel void findNeighbours (__global float3 *pos,
					   		  __global size_t *neighbours,
					   		  struct fluidParams params)
{
	size_t gID;
	size_t gSize;
	size_t nNeighbours;
	float dist;

	gID = get_global_id(0);
	gSize = get_global_size(0);
	nNeighbours = 0;

	for (size_t i = 0; i < gSize && nNeighbours < MAX_NEIGHBOURS; i++)
	{
		dist = distance(pos[gID], pos[i]);
		if (isless(dist, 2.0f * params.interactionRadius))
		{
			neighbours[gID * (MAX_NEIGHBOURS + 1) + ++nNeighbours] = i;
		}
	}

	neighbours[gID * (MAX_NEIGHBOURS + 1)] = nNeighbours;
}

float density(__global float3 *pos,
			  size_t gID,
			  __global size_t *neighbours,
			  size_t nNeighbours,
			  struct fluidParams params)
{
	float sum;
	sum = 0.0f;

	for (size_t i = 0; i < nNeighbours; i++)
	{
		sum += weightMonaghanSpline(pos[gID] - pos[neighbours[i]], params.interactionRadius);
	}

	sum *= params.monaghanSplineNormalisation * params.particleMass;

	return sum;
}

__kernel void updateDensity(__global float3 *pos,
						    __global float *densities,
					   		__global size_t *neighbours,
							struct fluidParams params)
{
	size_t gID;
	size_t gSize;

	gID = get_global_id(0);
	gSize = get_global_size(0);

	densities[gID] = density(pos, gID, &(neighbours[gID * (MAX_NEIGHBOURS + 1) + 1]), neighbours[gID * (MAX_NEIGHBOURS + 1)], params);
}

__kernel void updateNormals(__global float3 *pos,
							__global float *densities,
							__global float3 *normals,
							__global size_t *neighbours,
							struct fluidParams params)
{
	size_t gID;
	size_t gSize;
	size_t nNeighbours;

	float3 normal;

	gID = get_global_id(0);
	gSize = get_global_size(0);
	nNeighbours = neighbours[gID * (MAX_NEIGHBOURS + 1)];

	normal = 0.0f;

	size_t j;
	for (size_t i = 0; i < nNeighbours; i++)
	{
		j = neighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
	
		if (all(isequal(pos[gID], pos[j])))
		{
			continue;
		}
		normal += weightMonaghanSplinePrime(pos[gID] - pos[j], params.interactionRadius) * fast_normalize(pos[gID] - pos[j]) 
			      / densities[j];
	}

	normal *= params.interactionRadius * params.particleMass * params.monaghanSplinePrimeNormalisation;

	normals[gID] = normal;
}

__kernel void nonPressureForces(__global float3 *pos,
						  	    __global float3 *vel,
						  	    __global float3 *nonPressureForces,
						  	    __global float *densities,
						  	    __global float3 *normals,
					   			__global size_t *neighbours,
						  	    struct fluidParams params)
{
	size_t gID;
	size_t gSize;	
	size_t nNeighbours;
	float3 viscosityForce;
	float3 cohesionForce;
	float3 curvatureForce;

	gID = get_global_id(0);
	gSize = get_global_size(0);
	viscosityForce = 0.0f;
	cohesionForce = 0.0f;
	curvatureForce = 0.0f;

	nNeighbours = neighbours[gID * (MAX_NEIGHBOURS + 1)];
	
//	curvatureForce = normals[gID] * nNeighbours;

	size_t j;
	float surfaceTensionFactor;
	float3 r;
	float3 direction;
	for (size_t i = 0; i < nNeighbours; i++)
	{
		j = neighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		r = pos[gID] - pos[j];
		direction = normalize(r);

		surfaceTensionFactor = 2.0f * params.restDensity / (densities[gID] + densities[j]);

		/* Viscosity */
		viscosityForce += viscosityMonaghan(r, 
								   vel[gID] - vel[j], 
								   (densities[gID] + densities[j]) * 0.5f, 
								   params.interactionRadius)
			   			* weightMonaghanSplinePrime(r, params.interactionRadius)
			   			* direction;
		
		/* Surface tension cohesion */
		cohesionForce += surfaceTensionFactor * weightSurfaceTension(r, params.interactionRadius, params.surfaceTensionTerm) * direction;

		/* Surface tension curvature */
		curvatureForce += surfaceTensionFactor * normals[gID] - normals[j];

	}	

	viscosityForce *= -params.monaghanSplinePrimeNormalisation * params.viscosity * params.particleMass;
	cohesionForce *= -params.surfaceTensionNormalisation * params.surfaceTension * pown(params.particleMass, 2);
	curvatureForce *= -params.surfaceTension * params.particleMass;;

	nonPressureForces[gID] = viscosityForce + cohesionForce + curvatureForce + params.gravity * params.particleMass;
}

__kernel void initPressure(__global float4 *pressure)
{
	pressure[get_global_id(0)] = 0.0f;
}

__kernel void verletStep(__global float3 *posIn,
						 __global float3 *velIn,
						 __global float3 *posOut,
						 __global float3 *velOut,
						 __global float3 *nonPressureForces,
						 __global float4 *pressure,
						 __global float3 *oldForces,
						 struct fluidParams params)
{
	size_t gID;
	float3 force;
	gID = get_global_id(0);

	force = nonPressureForces[gID] + pressure[gID].xyz;
//TODO: sort out actual verlet integration
	velOut[gID] = velIn[gID] + params.timeStep * force / params.particleMass;
	posOut[gID] = mad(velIn[gID], params.timeStep, posIn[gID]) + 0.5f * force * pown(params.timeStep, 2) / params.particleMass;


//	posOut[gID] = posIn[gID] + params.timeStep * fma(params.timeStep, oldForces[gID] / params.particleMass, velIn[gID]);
//	velOut[gID] = velIn[gID] + params.timeStep * 0.5f * (oldForces[gID] + nonPressureForces[gID] + pressure[gID].xyz) / params.particleMass;

	/* Semi-Elastic Boundary */
	if (isgreater(posOut[gID].x, 0.6f))
	{
		posOut[gID].x = 1.2f - posOut[gID].x;
	 	velOut[gID].x *= -0.5f;
		velOut[gID].y *= 0.99f;
		velOut[gID].z *= 0.99f;
	}
	else if (isless(posOut[gID].x, -0.6f))
	{
		posOut[gID].x = -1.2f - posOut[gID].x;
		velOut[gID].x *= -0.5f;
		velOut[gID].y *= 0.99f;
		velOut[gID].z *= 0.99f;
	}

	if (isgreater(posOut[gID].y, 0.6f))
	{
		posOut[gID].y = 1.2f - posOut[gID].y;
		velOut[gID].x *= 0.99f;
		velOut[gID].y *= -0.5f;
		velOut[gID].z *= 0.99f;
	}
	else if (isless(posOut[gID].y, -0.6f))
	{
		posOut[gID].y = -1.2f - posOut[gID].y;
		velOut[gID].x *= 0.99f;
		velOut[gID].y *= -0.5f;
		velOut[gID].z *= 0.99f;
	}

	if (isgreater(posOut[gID].z, 1.2f))
	{
		posOut[gID].z = 2.4f - posOut[gID].z;
		velOut[gID].x *= 0.99f;
		velOut[gID].y *= 0.99f;
		velOut[gID].z *= -0.5f;
	}
	else if (isless(posOut[gID].z, -0.6f))
	{
		posOut[gID].z = -1.2f - posOut[gID].z;
		velOut[gID].x *= 0.99f;
		velOut[gID].y *= 0.99f;
		velOut[gID].z *= -0.5f;
	}

	posOut[gID] = clamp(posOut[gID], -10.0f, 10.0f);

}

__kernel void predictDensityPressure(__global float3 *pos,
									 __global float *densityErrors,
									 __global float4 *pressure, 
					   				 __global size_t *neighbours,
									 struct fluidParams params)
{
	size_t gID;
	size_t gSize;
	size_t nNeighbours;
	
	gID = get_global_id(0);
	gSize = get_global_size(0);

	nNeighbours = neighbours[gID * (MAX_NEIGHBOURS + 1)];

	/* Predict density error */
	densityErrors[gID] = max(0.0f, density(pos, gID, &(neighbours[gID * (MAX_NEIGHBOURS + 1) + 1]), nNeighbours, params) - params.restDensity);

	/* Update pressure */
	pressure[gID].w += densityErrors[gID] * params.pressureScalingFactor;
}

__kernel void pressureForces(__global float3 *pos,
							 __global float *densities,
						     __global float4 *pressure,
					   		 __global size_t *neighbours,
							 struct fluidParams params)
{
	size_t gID;
	size_t gSize;
	size_t nNeighbours;
	float pRhoSqr;
	float3 pressureForce;
	
	gID = get_global_id(0);
	gSize = get_global_size(0);

	nNeighbours = neighbours[gID * (MAX_NEIGHBOURS + 1)];

	pRhoSqr = pressure[gID].w / pown(densities[gID], 2);
	pressureForce = 0.0f;
	
	size_t j;
	for (size_t i = 0; i < nNeighbours; i++)
	{
		j = neighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		if (j == gID)
			continue;
		pressureForce += mad(pressure[j].w, pown(densities[j], -2), pRhoSqr) * weightMonaghanSplinePrime(pos[gID] - pos[j], params.interactionRadius) * fast_normalize(pos[gID] - pos[j]);
	}

	pressureForce *= -params.monaghanSplinePrimeNormalisation * pown(params.particleMass, 2);

	pressure[gID].xyz = pressureForce;

//	printf("pressure.w: %f\n", pressure[gID].w);
}

__kernel void updateOldForces(__global float3 *nonPressureForces,
							  __global float4 *pressure,
							  __global float3 *oldForces)
{
	size_t gID;
	gID = get_global_id(0);

	oldForces[gID] = nonPressureForces[gID] + pressure[gID].xyz;
}

float weightMonaghanSpline(float3 p12, float h)
{
	float q;
	float w;

	q = length(p12) / h;
	w = 0.0f;

	if (islessequal(q, 1.0f))
	{
		w += mad(q * q, mad(q, 0.75f, -1.5f), 1.0f);
	}
	else if (islessequal(q, 2.0f))
	{
		w += 0.25f * pow(2.0f - q, 3.0f);
	}

	return w;
}

float weightMonaghanSplinePrime(float3 p12, float h)
{
	float r;
	float w;

	r = fast_length(p12);
	w = 0.0f;

	if (islessequal(r, h))
	{
		w += r * mad(r, 3.0f, -4.0f * h);
	}
	else if (islessequal(r, 2.0f * h))
	{
		w += -pown(mad(-2.0f, h, r), 2);
	}
		
	w *= 0.75f * pown(h, -2);

	return w;
}

float weightSurfaceTension(float3 p12, float h, float extraTerm)
{
	float r;
	float w;

	r = fast_length(p12);
	w = 0.0f;

	if (islessequal(r, 0.001f * h))
	{
		return w;	
	}

	if (islessequal(r, h))
	{
		w += pown((h - r) * r, 3);
	}
	if (islessequal(2.0f * r, h))
	{
		w = mad(2.0f, w, extraTerm);
	}

	return w;
}


// v12:     v1 - v2
// p12:     p1 - p2
// rhoAvg:  average of (rho1, rho2)
float viscosityMonaghan(float3 p12,
						float3 v12,
						float  rhoAvg,
						float  interactionRadius)
{
	float vd = dot(v12, p12);
	
	if (isgreater(vd, 0.0f))
	{
		return 0.0f;
	}
	
	// Arbitrary terms - see Monaghan 1992
	float alpha, beta, etaSqr;
	alpha  = 1;
	beta   = 2;
	etaSqr = 0.01 * interactionRadius * interactionRadius;

	// Take c to be 1400 m/s for now
	float c;
	c = 1400;


	float mu12;
	mu12 = (interactionRadius * vd) / (dot(p12, p12) + etaSqr);

	return mu12 * mad(-alpha, c, beta * mu12) / rhoAvg;	
}

