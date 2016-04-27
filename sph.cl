#define OPENCL_COMPILING
#include "fluidParams.h"

#define MAX_NEIGHBOURS 200

float3 weightMonaghanSpline(float3 p12, float interactionRadius);

float3 weightMonaghanSplinePrime(float3 p12, float interactionRadius);

float3 viscosityMonaghan(float3 p12,
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
		if (isless(dist, 2.0f * interactionRadius) && gID != i)
		{
			neighbours[nNeighbours++] = i;
		}
	}

	return nNeighbours;
}

void density(__global float3 *pos,
		     __global float *densities,
			 size_t gID,
			 __local size_t *neighbours,
			 size_t nNeighbours,
			 struct fluidParams params)
{
	float sum;
	sum = 0;

	for (size_t i = 0; i < nNeighbours; i++)
	{
		sum += params.particleMass * length(weightMonaghanSpline(pos[gID] - pos[neighbours[i]], params.interactionRadius));
	}

	densities[gID] = sum;
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

	density(pos, densities, gID, neighbours, nNeighbours, params);
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
/*
	// Viscosity force
	for (size_t i = 0; i < nNeighbours; i++)
	{
//		force += (vel[neighbours[i]] - vel[gID]) * (distance(pos[gID], pos[neighbours[i]]) / densities[neighbours[i]]);
			force += viscosityMonaghan(pos[gID] - pos[neighbours[i]], 
									   vel[gID] - vel[neighbours[i]], 
									   (densities[gID] + densities[neighbours[i]]) * 0.5f, 
									   params.interactionRadius);
	}	
*/
	force *= params.viscosity;

	force += params.gravity;

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
	gID = get_global_id(0);

	velOut[gID] = velIn[gID] + params.timeStep * 0.5f * (oldForces[gID] + nonPressureForces[gID] + pressure[gID].xyz) / params.particleMass;
	posOut[gID] = posIn[gID] + params.timeStep * fma(params.timeStep, oldForces[gID] / params.particleMass, velOut[gID]);

	/* Elastic Boundary */
	if (isgreater(posOut[gID].x, 2.0f))
	{
		posOut[gID].x = 4.0f - posOut[gID].x;
		velOut[gID].x *= -1;
	}
	else if (isless(posOut[gID].x, -2.0f))
	{
		posOut[gID].x = -4.0f - posOut[gID].x;
		velOut[gID].x *= -1;
	}

	if (isgreater(posOut[gID].y, 2.0f))
	{
		posOut[gID].y = 4.0f - posOut[gID].y;
		velOut[gID].y *= -1;
	}
	else if (isless(posOut[gID].y, -2.0f))
	{
		posOut[gID].y = -4.0f - posOut[gID].y;
		velOut[gID].y *= -1;
	}

	if (isgreater(posOut[gID].z, 2.0f))
	{
		posOut[gID].z = 4.0f - posOut[gID].z;
		velOut[gID].z *= -1;
	}
	else if (isless(posOut[gID].z, -2.0f))
	{
		posOut[gID].z = -4.0f - posOut[gID].z;
		velOut[gID].z *= -1;
	}

//	posOut[gID] = clamp(posOut[gID], -2.0f, 2.0f);
}

__kernel void predictDensityPressure(__global float3 *pos,
									 __global float *densities,
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
	density(pos, densities, gID, neighbours, nNeighbours, params);
	densityErrors[gID] = densities[gID] - params.restDensity;

	/* Update pressure */
	pressure[gID].w -= densityErrors[gID] * params.pressureScalingFactor;
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
		pressureForce -= (pRhoSqr + pressure[j].w * pown(densities[j], -2)) * weightMonaghanSplinePrime(pos[gID] - pos[j], params.interactionRadius);
	}

	pressureForce *= -pown(params.particleMass, 2);

	pressure[gID].xyz = pressureForce;
}

__kernel void updateOldForces(__global float3 *nonPressureForces,
							  __global float4 *pressure,
							  __global float3 *oldForces)
{
	size_t gID;
	gID = get_global_id(0);

	oldForces[gID] = nonPressureForces[gID] + pressure[gID].xyz;
}

float3 weightMonaghanSpline(float3 p12, float interactionRadius)
{
	float3 q;
	float qD;
	float3 w;

	q = p12 / interactionRadius;
	qD = length(q);
	w = 0;

	if (isgreaterequal(qD, 0.0001f))
	{
		if (islessequal(qD, 1.0f))
		{
			w += 1.0f + q * q * fma(q, 0.75f, -1.5f);
		}
		else if (islessequal(qD, 2.0f))
		{
			w += 0.25f * pow(2.0f - q, 3.0f);
		}
	}
	
	w *= M_1_PI_F * pow(interactionRadius, -3.0f);

	return w;
}

float3 weightMonaghanSplinePrime(float3 p12, float interactionRadius)
{
	float q;
	float w;

	q = length(p12) / interactionRadius;
	w = 0.0f;

	if (isgreaterequal(q, 0.0001f))
	{
		if (islessequal(q, 1.0f))
		{
			w += 0.75f * q * fma(q, 3.0f, -4.0f);
		}
		else if (islessequal(q, 2.0f))
		{
			w += -0.75f * pown(2.0f - q, 2);
		}
	}
	
	w *= M_1_PI_F * pow(interactionRadius, -3.0f);

	return w * normalize(p12);
}

// v12:     v1 - v2
// p12:     p1 - p2
// rhoAvg:  average of (rho1, rho2)
float3 viscosityMonaghan(float3 p12,
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
	c = 1;


	float mu12;
	mu12 = (interactionRadius * vd) / (dot(p12, p12) + etaSqr);

	return normalize(p12) * mu12 * fma(-alpha, c, beta * mu12) / rhoAvg;	
}

