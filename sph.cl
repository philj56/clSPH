#define OPENCL_COMPILING
#include "fluidParams.h"

#define MAX_NEIGHBOURS 10000

float weightMonaghanSpline(float3 p12, float interactionRadius);

float weightMonaghanSplinePrime(float3 p12, float interactionRadius);

float viscosityMonaghan(float3 p12,
						float3 v12,
						float  rhoAvg,
						float  interactionRadius);

/* Simple neighbour search. TODO: optimise so neighbours aren't repeatedly found */
size_t findNeighbours (__global float3 *pos,
					   size_t gID,
					   size_t gSize,
					   __local size_t *neighbours,
					   size_t maxNeighbours,
					   float interactionRadius)
{
	size_t nNeighbours;
	float dist;
	nNeighbours = 0;
	for (size_t i = 0; i < gSize && nNeighbours < maxNeighbours; i++)
	{
		dist = distance(pos[gID], pos[i]);
		if (isless(dist, 2.0f * interactionRadius))
		{
			neighbours[nNeighbours++] = i;
		}
	}

//	printf("%u\n", nNeighbours);
	return nNeighbours;
}

float density(__global float3 *pos,
			  size_t gID,
			  __local size_t *neighbours,
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
							struct fluidParams params)
{
	size_t gID;
	size_t gSize;
	__local size_t neighbours[MAX_NEIGHBOURS];
	size_t nNeighbours;

	gID = get_global_id(0);
	gSize = get_global_size(0);
	nNeighbours = findNeighbours(pos, gID, gSize, neighbours, MAX_NEIGHBOURS, params.interactionRadius);

	densities[gID] = density(pos, gID, neighbours, nNeighbours, params);
}

__kernel void nonPressureForces(__global float3 *pos,
						  	    __global float3 *vel,
						  	    __global float3 *nonPressureForces,
						  	    __global float *densities,
						  	    struct fluidParams params)
{
	size_t gID;
	size_t gSize;	
	__local size_t neighbours [MAX_NEIGHBOURS];
	size_t nNeighbours;
	float3 force;

	gID = get_global_id(0);
	gSize = get_global_size(0);
	force = 0.0f;

	nNeighbours = findNeighbours (pos, gID, gSize, neighbours, MAX_NEIGHBOURS, params.interactionRadius);

	// Viscosity force
	for (size_t i = 0; i < nNeighbours; i++)
	{
		size_t j = neighbours[i];
//		force += (vel[neighbours[i]] - vel[gID]) * (weightMonaghanSpline(pos[gID], pos[neighbours[i]]) * densities[neighbours[i]]);
			force += viscosityMonaghan(pos[gID] - pos[j], 
									   vel[gID] - vel[j], 
									   (densities[gID] + densities[j]) * 0.5f, 
									   params.interactionRadius)
				   * weightMonaghanSplinePrime(pos[gID] - pos[j], params.interactionRadius)
				   * normalize(pos[gID] - pos[j]);
	}	

	force *= -params.monaghanSplinePrimeNormalisation * params.viscosity * params.particleMass;

	force += params.gravity * params.particleMass;

	nonPressureForces[gID] = force;
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
	posOut[gID] = posIn[gID] + velIn[gID] * params.timeStep + 0.5f * force * pown(params.timeStep, 2) / params.particleMass;


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

	if (isgreater(posOut[gID].z, 0.6f))
	{
		posOut[gID].z = 1.2f - posOut[gID].z;
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
									 struct fluidParams params)
{
	size_t gID;
	size_t gSize;
	__local size_t neighbours[MAX_NEIGHBOURS];
	size_t nNeighbours;
	
	gID = get_global_id(0);
	gSize = get_global_size(0);

	nNeighbours = findNeighbours (pos, gID, gSize, neighbours, MAX_NEIGHBOURS, params.interactionRadius);

	/* Predict density error */
	densityErrors[gID] = max(0.0f, density(pos, gID, neighbours, nNeighbours, params) - params.restDensity);

	/* Update pressure */
	pressure[gID].w += densityErrors[gID] * params.pressureScalingFactor;
}

__kernel void pressureForces(__global float3 *pos,
							 __global float *densities,
						     __global float4 *pressure,
							 struct fluidParams params)
{
	size_t gID;
	size_t gSize;
	__local size_t neighbours[MAX_NEIGHBOURS];
	size_t nNeighbours;
	float pRhoSqr;
	float3 pressureForce;
	
	gID = get_global_id(0);
	gSize = get_global_size(0);

	nNeighbours = findNeighbours (pos, gID, gSize, neighbours, MAX_NEIGHBOURS, params.interactionRadius);

	pRhoSqr = pressure[gID].w / pown(densities[gID], 2);
	pressureForce = 0.0f;
	
	size_t j;
	for (size_t i = 0; i < nNeighbours; i++)
	{
		j = neighbours[i];
		if (j == gID)
			continue;
		pressureForce += (pRhoSqr + pressure[j].w * pown(densities[j], -2)) * weightMonaghanSplinePrime(pos[gID] - pos[j], params.interactionRadius) * fast_normalize(pos[gID] - pos[j]);
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

float weightMonaghanSpline(float3 p12, float interactionRadius)
{
	float q;
	float w;

	q = length(p12) / interactionRadius;
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

float weightMonaghanSplinePrime(float3 p12, float interactionRadius)
{
	float r;
	float w;

	r = fast_length(p12);
	w = 0.0f;

	if (islessequal(r, 0.001f*interactionRadius))
		return w;

	if (islessequal(r, interactionRadius))
	{
		w += r * mad(r, 3.0f, -4.0f * interactionRadius);
	}
	else if (islessequal(r, 2.0f * interactionRadius))
	{
		w += -pown(mad(-2.0f, interactionRadius, r), 2);
	}
		
	w *= 0.75f * pown(interactionRadius, -2);

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

	return mu12 * fma(-alpha, c, beta * mu12) / rhoAvg;	
}

