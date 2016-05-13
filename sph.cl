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
__kernel void findNeighbours (__global __read_only const struct particle *particles,
							  __global __read_only const struct boundaryParticle *boundaryParticles,
					   		  __global __write_only size_t *neighbours,
					   		  __global __write_only size_t *boundaryNeighbours,
							  const size_t nBoundaryParticles,
					   		  const struct fluidParams params)
{
	size_t gID;
	size_t gSize;
	size_t nNeighbours;
	size_t nBoundaryNeighbours;
	float dist;
	struct particle self;

	gID = get_global_id(0);
	gSize = get_global_size(0);
	nNeighbours = 0;
	nBoundaryNeighbours = 0;

	self = particles[gID];

	for (size_t i = 0; i < gSize && nNeighbours < MAX_NEIGHBOURS; i++)
	{
		dist = distance(self.pos, particles[i].pos);
		if (isless(dist, self.interactionRadius) && i != gID)
		{
			nNeighbours++;
			neighbours[gID * (MAX_NEIGHBOURS + 1) + nNeighbours] = i;
		}
	}
	
	for (size_t i = 0; i < nBoundaryParticles && nBoundaryNeighbours < MAX_NEIGHBOURS; i++)
	{
		dist = distance(self.pos, boundaryParticles[i].pos);
		if (isless(dist, self.interactionRadius))
		{
			nBoundaryNeighbours++;
			boundaryNeighbours[gID * (MAX_NEIGHBOURS + 1) + nBoundaryNeighbours] = i;
		}
	}

	neighbours[gID * (MAX_NEIGHBOURS + 1)] = nNeighbours;
	boundaryNeighbours[gID * (MAX_NEIGHBOURS + 1)] = nBoundaryNeighbours;
}

__kernel void updateBoundaryVolumes(__global struct boundaryParticle *particles)
{
	size_t gID;
	size_t gSize;
	float sum;
	struct boundaryParticle self;

	gID = get_global_id(0);
	gSize = get_global_size(0);
	sum = 0.0f;

	self = particles[gID];

	for (size_t i = 0; i < gSize; i++)
	{
		if (distance(self.pos, particles[i].pos) <= self.interactionRadius)
			sum += weightMonaghanSpline(self.pos - particles[i].pos, self.interactionRadius);
	}

	particles[gID].volume = 1.0f / (sum * self.monaghanSplineNormalisation);
}

__kernel void updateDensity(__global struct particle *particles,
							__global __read_only const struct boundaryParticle *boundaryParticles,
					   		__global __read_only const size_t *neighbours,
					   		__global __read_only const size_t *boundaryNeighbours)
{
	size_t gID;
	size_t nNeighbours;
	size_t nBoundaryNeighbours;
	float density = 1.0f;
	struct particle self;

	gID = get_global_id(0);
	nNeighbours = neighbours[gID * (MAX_NEIGHBOURS + 1)];
	nBoundaryNeighbours = boundaryNeighbours[gID * (MAX_NEIGHBOURS + 1)];
	
	self = particles[gID];
	
	size_t j;
	struct particle other;
	struct boundaryParticle boundary;

	/* Fluid particles */
	for (size_t i = 0; i < nNeighbours; i++)
	{
		j = neighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		other = particles[j];

		density += weightMonaghanSpline(self.pos - other.pos, self.interactionRadius);
	}

	density *= self.mass;
	
	/* Boundary particles */
	for (size_t i = 0; i < nBoundaryNeighbours; i++)
	{
		j = boundaryNeighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		boundary = boundaryParticles[j];

		density += self.restDensity * boundary.volume * weightMonaghanSpline(self.pos - boundary.pos, self.interactionRadius);
	}

	density *= self.monaghanSplineNormalisation;// * self.kernelCorrection;

	particles[gID].density = density;	
}

__kernel void correctKernel(__global struct particle *particles,
							__global __read_only const struct boundaryParticle *boundaryParticles,
					   		__global __read_only const size_t *neighbours,
					   		__global __read_only const size_t *boundaryNeighbours)
{
	size_t gID;
	size_t nNeighbours;
	size_t nBoundaryNeighbours;
	float density = 1.0f;
	float correction;
	struct particle self;
	
	gID = get_global_id(0);;
	nNeighbours = neighbours[gID * (MAX_NEIGHBOURS + 1)];
	nBoundaryNeighbours = boundaryNeighbours[gID * (MAX_NEIGHBOURS + 1)];
	
	correction = 0.0f;
	self = particles[gID];

	size_t j;
	struct particle other;
	struct boundaryParticle boundary;
	
	/* Fluid particles */
	for (size_t i = 0; i < nNeighbours; i++)
	{
		j = neighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		other = particles[j];

		correction += isequal(self.restDensity, other.restDensity) * other.mass * weightMonaghanSpline(self.pos - other.pos, self.interactionRadius) / other.density;
		density += self.mass * weightMonaghanSpline(self.pos - other.pos, self.interactionRadius);
	}

	correction += 1.0f / self.density;

	correction *= self.monaghanSplineNormalisation;
	density *= self.monaghanSplineNormalisation;
	
	/* Boundary particles */
	for (size_t i = 0; i < nBoundaryNeighbours; i++)
	{
		j = boundaryNeighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		boundary = boundaryParticles[j];

		correction += boundary.volume * weightMonaghanSpline(self.pos - boundary.pos, self.interactionRadius);
		density += self.restDensity * boundary.volume * weightMonaghanSpline(self.pos - boundary.pos, self.interactionRadius);
	}


	particles[gID].density = 1.0f / correction * density;

	/* Skip correction if there are less than 2 neighbours or more than 2 boundary neighbours */
	if (nNeighbours <= 2 && nBoundaryNeighbours <= 2)
	{
		particles[gID].density = density;
	}
//	particles[gID].kernelCorrection = 1.0f / correction;*/
}

__kernel void updateNormals(__global struct particle *particles,
							__global __read_only const size_t *neighbours)
{
	size_t gID;
	size_t nNeighbours;
	float3 normal;
	struct particle self;

	gID = get_global_id(0);
	nNeighbours = neighbours[gID * (MAX_NEIGHBOURS + 1)];
	normal = 0.0f;

	self = particles[gID];

	size_t j;
	struct particle other;
	for (size_t i = 0; i < nNeighbours; i++)
	{
		j = neighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		other = particles[j];
		
		normal += isequal(self.restDensity, other.restDensity) * self.mass * weightMonaghanSplinePrime(self.pos - other.pos, self.interactionRadius) * fast_normalize(self.pos - other.pos) 
			      / other.density;
	}

	normal *= self.interactionRadius * self.mass * self.monaghanSplinePrimeNormalisation;

	particles[gID].normal = normal;
}

__kernel void nonPressureForces(__global struct particle *particles,
								__global __read_only const struct boundaryParticle *boundaryParticles,
					   			__global __read_only const size_t *neighbours,
					   			__global __read_only const size_t *boundaryNeighbours,
						  	    const struct fluidParams params)
{
	size_t gID;
	size_t nNeighbours;
	size_t nBoundaryNeighbours;
	float3 viscosityForce;
	float3 cohesionForce;
	float3 curvatureForce;
	float3 adhesionForce;
	struct particle self;

	gID = get_global_id(0);
	viscosityForce = 0.0f;
	cohesionForce = 0.0f;
	curvatureForce = 0.0f;
	adhesionForce = 0.0f;

	nNeighbours = neighbours[gID * (MAX_NEIGHBOURS + 1)];
	nBoundaryNeighbours = boundaryNeighbours[gID * (MAX_NEIGHBOURS + 1)];
	
	self = particles[gID];

	size_t j;
	float surfaceTensionFactor;
	float3 r;
	float3 direction;
	struct particle other;
	struct boundaryParticle boundary;

	/* Fluid particles */
	for (size_t i = 0; i < nNeighbours; i++)
	{
		j = neighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		other = particles[j];
		r = self.pos - other.pos;
		direction = normalize(r);

		surfaceTensionFactor = 2.0f * self.restDensity / (self.density + other.density);

		/* Viscosity */
		viscosityForce += viscosityMonaghan(r, 
								   self.vel - other.vel, 
								   (self.density + other.density) * 0.5f, 
								   self.interactionRadius)
			   			* weightMonaghanSplinePrime(r, self.interactionRadius)
			   			* direction * other.mass * self.viscosity;
		
		/* Surface tension cohesion */
		cohesionForce += isequal(self.restDensity, other.restDensity) * surfaceTensionFactor * other.mass * weightSurfaceTension(r, self.interactionRadius, self.surfaceTensionTerm) * direction;

		/* Surface tension curvature */
		curvatureForce += isequal(self.restDensity, other.restDensity) * surfaceTensionFactor * (self.normal - other.normal);

	}	
	
	/* Boundary particles */
	for (size_t i = 0; i < nBoundaryNeighbours; i++)
	{
		j = boundaryNeighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		boundary = boundaryParticles[j];
		r = self.pos - boundary.pos;
		direction = normalize(r);

		/* Viscosity */
		viscosityForce += viscosityMonaghan(r, 
								   self.vel, 
								   self.density * 2.0f, 
								   self.interactionRadius)
			   			* weightMonaghanSplinePrime(r, self.interactionRadius)
			   			* direction * self.restDensity * boundary.volume * boundary.viscosity;	

//		adhesionForce += self.restDensity * boundary.volume * weightAdhesion(r, self.interactionRadius) * direction * boundary.adhesionModifier;
	}	


	viscosityForce *= -self.monaghanSplinePrimeNormalisation * self.mass;
	cohesionForce *= -self.surfaceTensionNormalisation * self.surfaceTension * self.mass;
	curvatureForce *= -self.surfaceTension * self.mass;
	adhesionForce *= -self.adhesion * self.mass;

	particles[gID].velocityAdvection = self.vel + (viscosityForce + cohesionForce + curvatureForce + adhesionForce + params.gravity * self.mass) * params.timeStep / self.mass;
}

__kernel void initPressure(__global struct particle *particles,
						   __global __read_only const struct boundaryParticle *boundaryParticles,
					   	   __global __read_only const size_t *neighbours,
					   	   __global __read_only const size_t *boundaryNeighbours,
						   const struct fluidParams params)
{
	size_t gID;
	size_t nNeighbours;
	size_t nBoundaryNeighbours;

	float density;
	float aii;
	float3 dii;

	struct particle self;
	
	gID = get_global_id(0);
	nNeighbours = neighbours[gID * (MAX_NEIGHBOURS + 1)];
	nBoundaryNeighbours = boundaryNeighbours[gID * (MAX_NEIGHBOURS + 1)];

	density = 0.0f;
	aii = 0.0f;
	dii = 0.0f;

	self = particles[gID];

	size_t j;
	float weight;
	float3 direction;
	struct particle other;
	struct boundaryParticle boundary;

	/* Calculate density & dii */
	for (size_t i = 0; i < nNeighbours; i++)
	{
		j = neighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		other = particles[j];
		weight = weightMonaghanSplinePrime(self.pos - other.pos, self.interactionRadius);
		direction = normalize(self.pos - other.pos);

		dii += other.mass * weight * direction;
		density += self.mass * dot(self.velocityAdvection - other.velocityAdvection, weight * direction);
	}
	
	/* Boundary particles */
	for (size_t i = 0; i < nBoundaryNeighbours; i++)
	{
		j = boundaryNeighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		boundary = boundaryParticles[j];
		weight = weightMonaghanSplinePrime(self.pos - boundary.pos, self.interactionRadius);
		direction = normalize(self.pos - boundary.pos);

		dii += self.restDensity * boundary.volume * weight * direction;
		density += self.restDensity * boundary.volume * dot(self.velocityAdvection, weight * direction);
	}
	
	dii *= -pown(params.timeStep, 2) * pown(self.density, -2) * self.monaghanSplinePrimeNormalisation;
	density *= params.timeStep * self.monaghanSplinePrimeNormalisation;
	
	/* Predict density */
	self.densityAdvection = self.density + density;
	
	/* Initialise pressure */
	self.pressure *= 0.5f;

	float3 dji; 

	/* Calculate aii */
	for (size_t i = 0; i < nNeighbours; i++)
	{
		j = neighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		other = particles[j];
		weight = weightMonaghanSplinePrime(self.pos - other.pos, self.interactionRadius);
		direction = normalize(self.pos - other.pos);
		

		dji = self.mass * pown(params.timeStep, 2) * pown(self.density, -2) * weight * direction * self.monaghanSplinePrimeNormalisation;
		aii += other.mass * dot(dii - dji, weight * direction);
	}
	
	/* Boundary particles */
	for (size_t i = 0; i < nBoundaryNeighbours; i++)
	{
		j = boundaryNeighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		boundary = boundaryParticles[j];
		weight = weightMonaghanSplinePrime(self.pos - boundary.pos, self.interactionRadius);
		direction = normalize(self.pos - boundary.pos);

		dji = self.mass * pown(params.timeStep, 2) * pown(self.density, -2) * weight * direction * self.monaghanSplinePrimeNormalisation;
		aii += self.restDensity * boundary.volume * dot(dii - dji, weight * direction);
	}

	aii *= self.monaghanSplinePrimeNormalisation;

	/* Update coefficients */
	self.displacement = dii;
	self.advection = aii;

	particles[gID] = self;
}

__kernel void updateDij(__global struct particle *particles,
						__global __read_only const size_t *neighbours,
						const struct fluidParams params)
{
	size_t gID;
	size_t nNeighbours;

	float3 sum;

	struct particle self;

	gID = get_global_id(0);
	nNeighbours = neighbours[gID * (MAX_NEIGHBOURS + 1)];
	
	sum = 0.0f;

	self = particles[gID];

	size_t j;
	struct particle other;
	for (size_t i = 0; i < nNeighbours; i++)
	{
		j = neighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		other = particles[j];

		sum += other.mass * other.pressure * pown(other.density, -2) * weightMonaghanSplinePrime(self.pos - other.pos, self.interactionRadius) * normalize(self.pos - other.pos);
	}

	sum *= -pown(params.timeStep, 2) * self.monaghanSplinePrimeNormalisation;

	particles[gID].sumPressureMovement = sum;
}

__kernel void timeStep(__global struct particle *particles,
					   __global __read_only const float4 *force,
					   const struct fluidParams params)
{
	size_t gID;
	struct particle self;

	gID = get_global_id(0);
	self = particles[gID];

	self.vel = self.velocityAdvection + params.timeStep * force[gID].xyz / self.mass;
	self.pos += self.vel * params.timeStep;

	/* Semi-Elastic Boundary */
/*	if (isgreater(self.pos.x, 2.4f))
	{
		self.pos.x = 4.8f - self.pos.x;
	 	self.vel.x *= -0.5f;
		self.vel.y *= 0.99f;
		self.vel.z *= 0.99f;
	}
	else if (isless(self.pos.x, -0.6f))
	{
		self.pos.x = -1.2f - self.pos.x;
		self.vel.x *= -0.5f;
		self.vel.y *= 0.99f;
		self.vel.z *= 0.99f;
	}

	if (isgreater(self.pos.y, 0.6f))
	{
		self.pos.y = 1.2f - self.pos.y;
		self.vel.x *= 0.99f;
		self.vel.y *= -0.5f;
		self.vel.z *= 0.99f;
	}
	else if (isless(self.pos.y, -0.6f))
	{
		self.pos.y = -1.2f - self.pos.y;
		self.vel.x *= 0.99f;
		self.vel.y *= -0.5f;
		self.vel.z *= 0.99f;
	}
	if (isgreater(self.pos.z, 2.4f))
	{
		self.pos.z = 4.8f - self.pos.z;
		self.vel.x *= 0.99f;
		self.vel.y *= 0.99f;
		self.vel.z *= -0.5f;
	}
	else if (isless(self.pos.z, -0.6f))
	{
		self.pos.z = -1.2f - self.pos.z;
		self.vel.x *= 0.99f;
		self.vel.y *= 0.99f;
		self.vel.z *= -0.5f;
	}
		
*/	self.pos = clamp(self.pos, -10.0f, 10.0f);
	particles[gID].pos = self.pos;
	particles[gID].vel = self.vel;
}

__kernel void predictDensityPressure(__global struct particle *particles,
									 __global __read_only const struct boundaryParticle *boundaryParticles,
									 __global __write_only float *densityErrors,
									 __global __write_only float *pressureTemp, 
					   				 __global __read_only const size_t *neighbours,
					   				 __global __read_only const size_t *boundaryNeighbours,
	 								 const struct fluidParams params)
{
	size_t gID;
	size_t nNeighbours;
	size_t nBoundaryNeighbours;
	float factor;
	struct particle self;

	gID = get_global_id(0);
	nNeighbours = neighbours[gID * (MAX_NEIGHBOURS + 1)];
	nBoundaryNeighbours = boundaryNeighbours[gID * (MAX_NEIGHBOURS + 1)];
	factor = 0.0f;
	self = particles[gID];
	
	size_t j;
	float weight;
	float3 direction;
	struct particle other;
	struct boundaryParticle boundary;
	for (size_t i = 0; i < nNeighbours; i++)
	{
		j = neighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		other = particles[j];
		weight = weightMonaghanSplinePrime(self.pos - other.pos, self.interactionRadius);
		direction = normalize(self.pos - other.pos);

		factor += other.mass * dot(self.sumPressureMovement - other.displacement * other.pressure - other.sumPressureMovement 
		       + self.mass * pown(params.timeStep, 2) * pown(self.density, -2) * weight * self.monaghanSplinePrimeNormalisation 
			   * direction * self.pressure, weight*direction);
	}

	for (size_t i = 0; i < nBoundaryNeighbours; i++)
	{
		j = boundaryNeighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		boundary = boundaryParticles[j];
		weight = weightMonaghanSplinePrime(self.pos - boundary.pos, self.interactionRadius);
		direction = normalize(self.pos - boundary.pos);

		factor += self.restDensity * boundary.volume * dot(self.sumPressureMovement, weight*direction);
	}

	factor *= self.monaghanSplinePrimeNormalisation;	

	float temp;
	temp = 0.0f;

	/* Update pressure */
	if(self.advection != 0.0f)
		temp = (1.0f - params.relaxationFactor) * self.pressure + (1.0f * params.relaxationFactor / self.advection) * (self.restDensity - self.densityAdvection - factor);
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
	densityErrors[gID] = (self.densityAdvection + temp * self.advection + factor - self.restDensity) / self.restDensity;
}

__kernel void copyPressure(__global __read_only const float *in,
						   __global __write_only struct particle *particles,
						   __global __write_only float4 *out)
{
	size_t gID = get_global_id(0);

	out[gID].w = in[gID];
	particles[gID].pressure = in[gID];
}

__kernel void pressureForces(__global __read_only const struct particle *particles,
					   		 __global __read_only const struct boundaryParticle *boundaryParticles,
					   		 __global __read_only const size_t *neighbours,
					   		 __global __read_only const size_t *boundaryNeighbours,
							 __global __write_only float4 *pressure)
{
	size_t gID;
	size_t nNeighbours;
	size_t nBoundaryNeighbours;
	float pRhoSqr;
	float3 pressureForce;
	struct particle self;
	
	gID = get_global_id(0);

	nNeighbours = neighbours[gID * (MAX_NEIGHBOURS + 1)];
	nBoundaryNeighbours = boundaryNeighbours[gID * (MAX_NEIGHBOURS + 1)];
	self = particles[gID];

	pRhoSqr = self.pressure / pown(self.density, 2);
	pressureForce = 0.0f;

	
	size_t j;
	struct particle other;
	struct boundaryParticle boundary;
	
	/* Fluid particles */
	for (size_t i = 0; i < nNeighbours; i++)
	{
		j = neighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		other = particles[j];
//		pressureForce += other.mass * mad(other.pressure, pown(other.density, -2), pRhoSqr) * weightMonaghanSplinePrime(self.pos - other.pos, self.interactionRadius) * fast_normalize(self.pos - other.pos);
		pressureForce += other.mass * other.pressure / (self.density * other.density) * weightMonaghanSplinePrime(self.pos - other.pos, self.interactionRadius) * fast_normalize(self.pos - other.pos);
	}
	
	/* Boundary particles */
	for (size_t i = 0; i < nBoundaryNeighbours; i++)
	{
		j = boundaryNeighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		boundary = boundaryParticles[j];
		pressureForce += boundary.volume * self.restDensity * pRhoSqr * weightMonaghanSplinePrime(self.pos - boundary.pos, self.interactionRadius) * fast_normalize(self.pos - boundary.pos);
	}

	pressureForce *= -self.monaghanSplinePrimeNormalisation * self.mass;

	pressure[gID].xyz = pressureForce;
	
	if(gID == 925 && (self.pos.x > 9.0f || self.pos.x < -9.0f))
		printf("%u:\npos:\t%.16v3hlf\nvel:\t%.16v3hlf\ndensity:\t%.16f\npressure:\t%.16f\nadvec:\t%.16f\ndensityAdvec:\t%.16f\n", gID, 
				self.pos, self.vel, self.density, self.pressure, self.advection, self.densityAdvection);
}

float weightMonaghanSpline(float3 p12, float h)
{
	float q;
	h *= 0.5f;

	q = fast_length(p12) / h;

	return islessequal(q, 1.0f) * mad(q * q, mad(q, 0.75f, -1.5f), 1.0f) + 			/* q < 1 */
		   isgreater(q, 1.0f) * islessequal(q, 2.0f) * 0.25f * pow(2.0f - q, 3.0f);	/* 1 < q < 2 */
}

float weightMonaghanSplinePrime(float3 p12, float h)
{
	float r;
	h *= 0.5f;

	r = fast_length(p12);

	return 0.75f * pown(h, -2) * (islessequal(r, h) * r * mad(r, 3.0f, -4.0f * h)								/* r < 0.5h */
								  - isgreater(r, h) * islessequal(r, 2.0f * h) * pown(mad(-2.0f, h, r), 2));	/* 0.5h < r < h */
}

float weightSurfaceTension(float3 p12, float h, float extraTerm)
{
	float r;
	r = fast_length(p12);

	return isgreater(r, 0.001f * h) * islessequal(r, h) * pown((h - r) * r, 3) 			/* r < h */
		   * (1.0f + islessequal(2.0f * r, h)) + islessequal(2.0f * r, h) * extraTerm;	/* r < h/2 */
}

float weightAdhesion(float3 p12, float h)
{
	float r;
	r = fast_length(p12);

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
	float alpha, beta, etaSqr;
	alpha  = 1;
	beta   = 2;
	etaSqr = 0.01 * interactionRadius * interactionRadius;

	// Take c to be 1400 m/s for now
	float c;
	c = 1400;

	float mu12;
	mu12 = (interactionRadius * vd) / (dot(p12, p12) + etaSqr);

	return isless(vd, 0.0f) * mu12 * mad(-alpha, c, beta * mu12) / rhoAvg;	
}

