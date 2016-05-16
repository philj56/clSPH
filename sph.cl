#define OPENCL_COMPILING
#include "fluidParams.h"
#include "particle.h"

float weightMonaghanSpline(float3 p12, float h);

float weightMonaghanSplinePrime(float3 p12, float h);

float weightSurfaceTension(float3 p12, float h, float extraTerm);

float weightAdhesion(float3 p12, float h);

float viscosityMonaghan(float3 p12,
						float3 v12,
						float  rhoAvg,
						float  interactionRadius);

/* Simple neighbour search */
__kernel void findNeighbours (__global __read_only const particle_t *particles,
							  __global __read_only const particle_t *boundaryParticles,
					   		  __global __write_only size_t *neighbours,
					   		  __global __write_only size_t *boundaryNeighbours,
							  const size_t nBoundaryParticles,
					   		  __constant fluid_t *params)
{
	size_t gID = get_global_id(0);
	size_t gSize = get_global_size(0);
	size_t nNeighbours = 0;
	size_t nBoundaryNeighbours = 0;
	float dist;
	particle_t self = particles[gID];

	for (size_t i = 0; i < gSize && nNeighbours < MAX_NEIGHBOURS; i++)
	{
		dist = distance(self.pos, particles[i].pos);
		if (isless(dist, params[self.fluidIndex].interactionRadius) && i != gID)
		{
			nNeighbours++;
			neighbours[gID * (MAX_NEIGHBOURS + 1) + nNeighbours] = i;
		}
	}
	
	for (size_t i = 0; i < nBoundaryParticles && nBoundaryNeighbours < MAX_NEIGHBOURS; i++)
	{
		dist = distance(self.pos, boundaryParticles[i].pos);
		if (isless(dist, params[self.fluidIndex].interactionRadius))
		{
			nBoundaryNeighbours++;
			boundaryNeighbours[gID * (MAX_NEIGHBOURS + 1) + nBoundaryNeighbours] = i;
		}
	}

	neighbours[gID * (MAX_NEIGHBOURS + 1)] = nNeighbours;
	boundaryNeighbours[gID * (MAX_NEIGHBOURS + 1)] = nBoundaryNeighbours;
}

__kernel void updateBoundaryVolumes(__global particle_t *particles,
									__constant fluid_t *params)
{
	size_t gID = get_global_id(0);
	size_t gSize = get_global_size(0);
	float sum = 0.0f;
	particle_t self = particles[gID];

	for (size_t i = 0; i < gSize; i++)
	{
		if (distance(self.pos, particles[i].pos) <= params[self.fluidIndex].interactionRadius)
			sum += weightMonaghanSpline(self.pos - particles[i].pos, params[self.fluidIndex].interactionRadius);
	}

	particles[gID].volume = 1.0f / (sum * params[self.fluidIndex].monaghanSplineNormalisation);
}

__kernel void updateDensity(__global particle_t *particles,
							__global __read_only const particle_t *boundaryParticles,
					   		__global __read_only const size_t *neighbours,
					   		__global __read_only const size_t *boundaryNeighbours,
							__constant fluid_t *params)
{
	size_t gID = get_global_id(0);
	size_t nNeighbours = neighbours[gID * (MAX_NEIGHBOURS + 1)];
	size_t nBoundaryNeighbours = boundaryNeighbours[gID * (MAX_NEIGHBOURS + 1)];
	float density = 1.0f;
	particle_t self = particles[gID];
	
	size_t j;
	particle_t other;

	/* Fluid particles */
	for (size_t i = 0; i < nNeighbours; i++)
	{
		j = neighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		other = particles[j];

		density += weightMonaghanSpline(self.pos - other.pos, params[self.fluidIndex].interactionRadius);
	}

	density *= params[self.fluidIndex].mass;
	
	/* Boundary particles */
	for (size_t i = 0; i < nBoundaryNeighbours; i++)
	{
		j = boundaryNeighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		other = boundaryParticles[j];

		density += params[self.fluidIndex].restDensity * other.volume * weightMonaghanSpline(self.pos - other.pos, params[self.fluidIndex].interactionRadius);
	}

	density *= params[self.fluidIndex].monaghanSplineNormalisation;// * self.kernelCorrection;

	particles[gID].density = density;	
}

__kernel void correctKernel(__global particle_t *particles,
							__global __read_only const particle_t *boundaryParticles,
					   		__global __read_only const size_t *neighbours,
					   		__global __read_only const size_t *boundaryNeighbours,
							__constant fluid_t *params)
{
	size_t gID = get_global_id(0);
	size_t nNeighbours = neighbours[gID * (MAX_NEIGHBOURS + 1)];
	size_t nBoundaryNeighbours = boundaryNeighbours[gID * (MAX_NEIGHBOURS + 1)];
	float density = 1.0f;
	float correction = 0.0f;
	particle_t self = particles[gID];
	
	size_t j;
	particle_t other;
	
	/* Fluid particles */
	for (size_t i = 0; i < nNeighbours; i++)
	{
		j = neighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		other = particles[j];

		correction += (self.fluidIndex == other.fluidIndex) * params[self.fluidIndex].mass * weightMonaghanSpline(self.pos - other.pos, params[self.fluidIndex].interactionRadius) / other.density;
		density += params[self.fluidIndex].mass * weightMonaghanSpline(self.pos - other.pos, params[self.fluidIndex].interactionRadius);
	}

	correction += 1.0f / self.density;

	correction *= params[self.fluidIndex].monaghanSplineNormalisation;
	density *= params[self.fluidIndex].monaghanSplineNormalisation;
	
	/* Boundary particles */
	for (size_t i = 0; i < nBoundaryNeighbours; i++)
	{
		j = boundaryNeighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		other = boundaryParticles[j];

		correction += other.volume * weightMonaghanSpline(self.pos - other.pos, params[self.fluidIndex].interactionRadius);
		density += params[self.fluidIndex].restDensity * other.volume * weightMonaghanSpline(self.pos - other.pos, params[self.fluidIndex].interactionRadius);
	}


	particles[gID].density = 1.0f / correction * density;

	/* Skip correction if there are less than 2 neighbours or more than 2 boundary neighbours */
	if (nNeighbours <= 2 && nBoundaryNeighbours <= 2)
	{
		particles[gID].density = density;
	}
//	particles[gID].kernelCorrection = 1.0f / correction;*/
}

__kernel void updateNormals(__global particle_t *particles,
							__global __read_only const size_t *neighbours,
							__constant fluid_t *params)
{
	size_t gID = get_global_id(0);
	size_t nNeighbours = neighbours[gID * (MAX_NEIGHBOURS + 1)];
	float3 normal = 0.0f;
	particle_t self = particles[gID];

	size_t j;
	particle_t other;

	for (size_t i = 0; i < nNeighbours; i++)
	{
		j = neighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		other = particles[j];
		
		normal += (self.fluidIndex == other.fluidIndex) * params[self.fluidIndex].mass * weightMonaghanSplinePrime(self.pos - other.pos, params[self.fluidIndex].interactionRadius) * fast_normalize(self.pos - other.pos) 
			      / other.density;
	}

	normal *= params[self.fluidIndex].interactionRadius * params[self.fluidIndex].mass * params[self.fluidIndex].monaghanSplinePrimeNormalisation;

	particles[gID].normal = normal;
}

__kernel void nonPressureForces(__global particle_t *particles,
								__global __read_only const particle_t *boundaryParticles,
					   			__global __read_only const size_t *neighbours,
					   			__global __read_only const size_t *boundaryNeighbours,
						  	    __constant fluid_t *params)
{
	size_t gID = get_global_id(0);
	size_t nNeighbours = neighbours[gID * (MAX_NEIGHBOURS + 1)];
	size_t nBoundaryNeighbours = boundaryNeighbours[gID * (MAX_NEIGHBOURS + 1)];
	float3 viscosityForce = 0.0f;
	float3 cohesionForce = 0.0f;
	float3 curvatureForce = 0.0f;
	float3 adhesionForce = 0.0f;
	particle_t self = particles[gID];

	size_t j;
	float surfaceTensionFactor;
	float3 r;
	float3 direction;
	particle_t other;

	/* Fluid particles */
	for (size_t i = 0; i < nNeighbours; i++)
	{
		j = neighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		other = particles[j];
		r = self.pos - other.pos;
		direction = normalize(r);

		surfaceTensionFactor = 2.0f * params[self.fluidIndex].restDensity / (self.density + other.density);

		/* Viscosity */
		viscosityForce += viscosityMonaghan(r, 
								   self.vel - other.vel, 
								   (self.density + other.density) * 0.5f, 
								   params[self.fluidIndex].interactionRadius)
			   			* weightMonaghanSplinePrime(r, params[self.fluidIndex].interactionRadius)
			   			* direction * params[other.fluidIndex].mass * params[self.fluidIndex].viscosity;
		
		/* Surface tension cohesion */
		cohesionForce += (self.fluidIndex == other.fluidIndex) * surfaceTensionFactor * params[self.fluidIndex].mass 
				       * weightSurfaceTension(r, params[self.fluidIndex].interactionRadius, params[self.fluidIndex].surfaceTensionTerm) * direction;

		/* Surface tension curvature */
		curvatureForce += (self.fluidIndex == other.fluidIndex) * surfaceTensionFactor * (self.normal - other.normal);

	}	
	
	/* Boundary particles */
	for (size_t i = 0; i < nBoundaryNeighbours; i++)
	{
		j = boundaryNeighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		other = boundaryParticles[j];
		r = self.pos - other.pos;
		direction = normalize(r);

		/* Viscosity */
		viscosityForce += viscosityMonaghan(r, 
								   self.vel, 
								   self.density * 2.0f, 
								   params[self.fluidIndex].interactionRadius)
			   			* weightMonaghanSplinePrime(r, params[self.fluidIndex].interactionRadius)
			   			* direction * params[self.fluidIndex].restDensity * other.volume * params[other.fluidIndex].viscosity;	

//		adhesionForce += self.restDensity * boundary.volume * weightAdhesion(r, self.interactionRadius) * direction * boundary.adhesionModifier;
	}	


	viscosityForce *= -params[self.fluidIndex].monaghanSplinePrimeNormalisation * params[self.fluidIndex].mass;
	cohesionForce *= -params[self.fluidIndex].surfaceTensionNormalisation * params[self.fluidIndex].surfaceTension * params[self.fluidIndex].mass;
	curvatureForce *= -params[self.fluidIndex].surfaceTension * params[self.fluidIndex].mass;
	adhesionForce *= -params[self.fluidIndex].adhesion * params[self.fluidIndex].mass;

	particles[gID].velocityAdvection = self.vel + (viscosityForce + cohesionForce + curvatureForce + adhesionForce + params[self.fluidIndex].gravity * params[self.fluidIndex].mass) * params[self.fluidIndex].timeStep / params[self.fluidIndex].mass;
}

__kernel void initPressure(__global particle_t *particles,
						   __global __read_only const particle_t *boundaryParticles,
					   	   __global __read_only const size_t *neighbours,
					   	   __global __read_only const size_t *boundaryNeighbours,
						   __constant fluid_t *params)
{
	size_t gID = get_global_id(0);
	size_t nNeighbours = neighbours[gID * (MAX_NEIGHBOURS + 1)];
	size_t nBoundaryNeighbours = boundaryNeighbours[gID * (MAX_NEIGHBOURS + 1)];
	float density = 0.0f;
	float aii = 0.0f;
	float3 dii = 0.0f;
	particle_t self = particles[gID];
	
	size_t j;
	float weight;
	float3 direction;
	float3 dji; 
	particle_t other;

	/* Calculate density & dii */
	for (size_t i = 0; i < nNeighbours; i++)
	{
		j = neighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		other = particles[j];
		weight = weightMonaghanSplinePrime(self.pos - other.pos, params[self.fluidIndex].interactionRadius);
		direction = normalize(self.pos - other.pos);

		dii += params[other.fluidIndex].mass * weight * direction;
		density += params[self.fluidIndex].mass * dot(self.velocityAdvection - other.velocityAdvection, weight * direction);
	}
	
	/* Boundary particles */
	for (size_t i = 0; i < nBoundaryNeighbours; i++)
	{
		j = boundaryNeighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		other = boundaryParticles[j];
		weight = weightMonaghanSplinePrime(self.pos - other.pos, params[self.fluidIndex].interactionRadius);
		direction = normalize(self.pos - other.pos);

		dii += params[self.fluidIndex].restDensity * other.volume * weight * direction;
		density += params[self.fluidIndex].restDensity * other.volume * dot(self.velocityAdvection, weight * direction);
	}
	
	dii *= -pown(params[self.fluidIndex].timeStep, 2) * pown(self.density, -2) * params[self.fluidIndex].monaghanSplinePrimeNormalisation;
	density *= params[self.fluidIndex].timeStep * params[self.fluidIndex].monaghanSplinePrimeNormalisation;
	
	/* Predict density */
	self.densityAdvection = self.density + density;
	
	/* Initialise pressure */
	self.pressure *= 0.5f;

	/* Calculate aii */
	for (size_t i = 0; i < nNeighbours; i++)
	{
		j = neighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		other = particles[j];
		weight = weightMonaghanSplinePrime(self.pos - other.pos, params[self.fluidIndex].interactionRadius);
		direction = normalize(self.pos - other.pos);
		

		dji = params[self.fluidIndex].mass * pown(params[self.fluidIndex].timeStep, 2) * pown(self.density, -2) * weight * direction * params[self.fluidIndex].monaghanSplinePrimeNormalisation;
		aii += params[other.fluidIndex].mass * dot(dii - dji, weight * direction);
	}
	
	/* Boundary particles */
	for (size_t i = 0; i < nBoundaryNeighbours; i++)
	{
		j = boundaryNeighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		other = boundaryParticles[j];
		weight = weightMonaghanSplinePrime(self.pos - other.pos, params[self.fluidIndex].interactionRadius);
		direction = normalize(self.pos - other.pos);

		dji = params[self.fluidIndex].restDensity * other.volume * pown(params[self.fluidIndex].timeStep, 2) * pown(self.density, -2) * weight * direction * params[self.fluidIndex].monaghanSplinePrimeNormalisation;
		aii += params[self.fluidIndex].restDensity * other.volume * dot(dii - dji, weight * direction);
	}

	aii *= params[self.fluidIndex].monaghanSplinePrimeNormalisation;

	/* Update coefficients */
	self.displacement = dii;
	self.advection = aii;

	particles[gID] = self;
}

__kernel void updateDij(__global particle_t *particles,
						__global __read_only const size_t *neighbours,
						__constant fluid_t *params)
{
	size_t gID = get_global_id(0);
	size_t nNeighbours = neighbours[gID * (MAX_NEIGHBOURS + 1)];
	float3 sum = 0.0f;
	particle_t self = particles[gID];

	size_t j;
	particle_t other;

	for (size_t i = 0; i < nNeighbours; i++)
	{
		j = neighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		other = particles[j];

		sum += params[other.fluidIndex].mass * other.pressure * pown(other.density, -2) * weightMonaghanSplinePrime(self.pos - other.pos, params[self.fluidIndex].interactionRadius) * normalize(self.pos - other.pos);
	}

	sum *= -pown(params[self.fluidIndex].timeStep, 2) * params[self.fluidIndex].monaghanSplinePrimeNormalisation;

	particles[gID].sumPressureMovement = sum;
}

__kernel void timeStep(__global particle_t *particles,
					   __global __read_only const float4 *force,
					   __constant fluid_t *params)
{
	size_t gID = get_global_id(0);
	particle_t self = particles[gID];

	self.vel = self.velocityAdvection + params[self.fluidIndex].timeStep * force[gID].xyz / params[self.fluidIndex].mass;
	self.pos += self.vel * params[self.fluidIndex].timeStep;

	/* For now, clamp particle position to a range */
	particles[gID].pos = clamp(self.pos, -10.0f, 10.0f);
	particles[gID].vel = self.vel;
}

__kernel void predictDensityPressure(__global particle_t *particles,
									 __global __read_only const particle_t *boundaryParticles,
									 __global __write_only float *densityErrors,
									 __global __write_only float *pressureTemp, 
					   				 __global __read_only const size_t *neighbours,
					   				 __global __read_only const size_t *boundaryNeighbours,
	 								 __constant fluid_t *params)
{
	size_t gID = get_global_id(0);
	size_t nNeighbours = neighbours[gID * (MAX_NEIGHBOURS + 1)];
	size_t nBoundaryNeighbours = boundaryNeighbours[gID * (MAX_NEIGHBOURS + 1)];
	float factor = 0.0f;
	float temp = 0.0f;
	particle_t self = particles[gID];
	
	size_t j;
	float weight;
	float3 direction;
	particle_t other;

	/* Fluid particles */
	for (size_t i = 0; i < nNeighbours; i++)
	{
		j = neighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		other = particles[j];
		weight = weightMonaghanSplinePrime(self.pos - other.pos, params[self.fluidIndex].interactionRadius);
		direction = normalize(self.pos - other.pos);

		factor += params[other.fluidIndex].mass * dot(self.sumPressureMovement - other.displacement * other.pressure - other.sumPressureMovement 
		       + params[self.fluidIndex].mass * pown(params[self.fluidIndex].timeStep, 2) * pown(self.density, -2) * weight * params[self.fluidIndex].monaghanSplinePrimeNormalisation 
			   * direction * self.pressure, weight*direction);
	}

	/* Boundary particles */
	for (size_t i = 0; i < nBoundaryNeighbours; i++)
	{
		j = boundaryNeighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		other = boundaryParticles[j];
		weight = weightMonaghanSplinePrime(self.pos - other.pos, params[self.fluidIndex].interactionRadius);
		direction = normalize(self.pos - other.pos);

		factor += params[self.fluidIndex].restDensity * other.volume * dot(self.sumPressureMovement, weight*direction);
	}

	factor *= params[self.fluidIndex].monaghanSplinePrimeNormalisation;	

	/* Update pressure */
	if(self.advection != 0.0f)
		temp = (1.0f - params[self.fluidIndex].relaxationFactor) * self.pressure + (1.0f * params[self.fluidIndex].relaxationFactor / self.advection) * (params[self.fluidIndex].restDensity - self.densityAdvection - factor);
//	temp = max(temp, 0.0f);
	pressureTemp[gID] = max(temp, 0.0f);

/*	if (gID == 680)
	{
		printf("Advection:    %f\n", self.advection);
		printf("Density:      %f\n", self.density);
		printf("DensityAdvec: %f\n", self.densityAdvection);
		printf("Pressure:     %f\n", temp);
	}
*/
	/* Predict density error */
//	particles[gID].densityAdvection = self.densityAdvection + temp * self.advection + factor;
	densityErrors[gID] = (self.densityAdvection + temp * self.advection + factor - params[self.fluidIndex].restDensity) / params[self.fluidIndex].restDensity;
}

__kernel void copyPressure(__global __read_only const float *in,
						   __global __write_only particle_t *particles,
						   __global __write_only float4 *out)
{
	size_t gID = get_global_id(0);

	out[gID].w = in[gID];
	particles[gID].pressure = in[gID];
}

__kernel void pressureForces(__global __read_only const particle_t *particles,
					   		 __global __read_only const particle_t *boundaryParticles,
					   		 __global __read_only const size_t *neighbours,
					   		 __global __read_only const size_t *boundaryNeighbours,
							 __global __write_only float4 *pressure,
							 __constant fluid_t *params)
{
	size_t gID = get_global_id(0);
	size_t nNeighbours = neighbours[gID * (MAX_NEIGHBOURS + 1)];
	size_t nBoundaryNeighbours = boundaryNeighbours[gID * (MAX_NEIGHBOURS + 1)];
	float3 pressureForce = 0.0f;
	particle_t self = particles[gID];
	float pRhoSqr = self.pressure / pown(self.density, 2);
	
	size_t j;
	particle_t other;
	
	/* Fluid particles */
	for (size_t i = 0; i < nNeighbours; i++)
	{
		j = neighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		other = particles[j];
//		pressureForce += other.mass * mad(other.pressure, pown(other.density, -2), pRhoSqr) * weightMonaghanSplinePrime(self.pos - other.pos, self.interactionRadius) * fast_normalize(self.pos - other.pos);
		pressureForce += params[other.fluidIndex].mass * other.pressure / (self.density * other.density) * weightMonaghanSplinePrime(self.pos - other.pos, params[self.fluidIndex].interactionRadius) * fast_normalize(self.pos - other.pos);
	}
	
	/* Boundary particles */
	for (size_t i = 0; i < nBoundaryNeighbours; i++)
	{
		j = boundaryNeighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		other = boundaryParticles[j];
		pressureForce += other.volume * params[self.fluidIndex].restDensity * pRhoSqr * weightMonaghanSplinePrime(self.pos - other.pos, params[self.fluidIndex].interactionRadius) * fast_normalize(self.pos - other.pos);
	}

	pressureForce *= -params[self.fluidIndex].monaghanSplinePrimeNormalisation * params[self.fluidIndex].mass;

	pressure[gID].xyz = pressureForce;
	
	if(gID == 925 && (self.pos.x > 9.0f || self.pos.x < -9.0f))
		printf("%u:\npos:\t%.16v3hlf\nvel:\t%.16v3hlf\ndensity:\t%.16f\npressure:\t%.16f\nadvec:\t%.16f\ndensityAdvec:\t%.16f\n", gID, 
				self.pos, self.vel, self.density, self.pressure, self.advection, self.densityAdvection);
}

float weightMonaghanSpline(float3 p12, float h)
{
	float q = fast_length(p12) / (0.5f * h);

	return islessequal(q, 1.0f) * mad(q * q, mad(q, 0.75f, -1.5f), 1.0f) + 			/* q < 1 */
		   isgreater(q, 1.0f) * islessequal(q, 2.0f) * 0.25f * pow(2.0f - q, 3.0f);	/* 1 < q < 2 */
}

float weightMonaghanSplinePrime(float3 p12, float h)
{
	float r = fast_length(p12);
	h *= 0.5f;

	return 0.75f * pown(h, -2) * (islessequal(r, h) * r * mad(r, 3.0f, -4.0f * h)								/* r < 0.5h */
								  - isgreater(r, h) * islessequal(r, 2.0f * h) * pown(mad(-2.0f, h, r), 2));	/* 0.5h < r < h */
}

float weightSurfaceTension(float3 p12, float h, float extraTerm)
{
	float r = fast_length(p12);

	return isgreater(r, 0.001f * h) * islessequal(r, h) * pown((h - r) * r, 3) 			/* r < h */
		   * (1.0f + islessequal(2.0f * r, h)) + islessequal(2.0f * r, h) * extraTerm;	/* r < h/2 */
}

float weightAdhesion(float3 p12, float h)
{
	float r = fast_length(p12);

	return sqrt(sqrt(isgreaterequal(2.0f * r, h) * islessequal(r, h) * (- 4.0f * r * r / h + 6.0f * r - 2.0f * h )));	/* h/2 < r < h */
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
	
	// Arbitrary terms - see Monaghan 1992
	float alpha = 1;
	float beta = 2;
	float etaSqr = 0.01 * interactionRadius * interactionRadius;

	// Take c to be 1400 m/s for now
	float c = 1400;

	float mu12  = (interactionRadius * vd) / (dot(p12, p12) + etaSqr);

	return isless(vd, 0.0f) * mu12 * mad(-alpha, c, beta * mu12) / rhoAvg;	
}

