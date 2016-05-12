#define OPENCL_COMPILING
#include "fluidParams.h"
#include "particle.h"

float weightMonaghanSpline(float3 p12, float interactionRadius);

float weightMonaghanSplinePrime(float3 p12, float interactionRadius);

float weightSurfaceTension(float3 p12, float h, float extraTerm);

float viscosityMonaghan(float3 p12,
						float3 v12,
						float  rhoAvg,
						float  interactionRadius);

/* Simple neighbour search. TODO: optimise so neighbours aren't repeatedly found */
__kernel void findNeighbours (__global __read_only const struct particle *particles,
					   		  __global __write_only size_t *neighbours,
					   		  const struct fluidParams params)
{
	size_t gID;
	size_t gSize;
	size_t nNeighbours;
	float dist;
	struct particle self;

	gID = get_global_id(0);
	gSize = get_global_size(0);
	nNeighbours = 0;

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

	neighbours[gID * (MAX_NEIGHBOURS + 1)] = nNeighbours;
}

__kernel void updateDensity(__global struct particle *particles,
					   		__global __read_only const size_t *neighbours)
{
	size_t gID;
	size_t nNeighbours;
	float density = 1.0f;
	struct particle self;

	gID = get_global_id(0);
	nNeighbours = neighbours[gID * (MAX_NEIGHBOURS + 1)];
	
	self = particles[gID];
	
	size_t j;
	struct particle other;
	for (size_t i = 0; i < nNeighbours; i++)
	{
		j = neighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		other = particles[j];

		density += other.mass * weightMonaghanSpline(self.pos - other.pos, self.interactionRadius);
	}

	density *= self.monaghanSplineNormalisation * self.kernelCorrection;

	particles[gID].density = density;	
}

__kernel void correctKernel(__global struct particle *particles,
						    __global __read_only const size_t *neighbours)
{
	size_t gID;
	size_t nNeighbours;
	float density = 1.0f;
	float correction;
	struct particle self;
	
	gID = get_global_id(0);;
	nNeighbours = neighbours[gID * (MAX_NEIGHBOURS + 1)];

	if (nNeighbours <= 3)
	{
		particles[gID].kernelCorrection = 1.0f;
		return;
	}
	
	correction = 0.0f;
	self = particles[gID];

	size_t j;
	struct particle other;
	for (size_t i = 0; i < nNeighbours; i++)
	{
		j = neighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		other = particles[j];

		correction += other.mass * weightMonaghanSpline(self.pos - other.pos, self.interactionRadius) / other.density;
		density += other.mass * weightMonaghanSpline(self.pos - other.pos, self.interactionRadius);
	}

	correction += 1.0f / self.density;

	correction *= self.monaghanSplineNormalisation;
	density *= self.monaghanSplineNormalisation;

	particles[gID].density = 1.0f / correction * density;
//	particles[gID].kernelCorrection = 1.0f / correction;
}

__kernel void updateNormals(__global struct particle *particles,
							__global __read_only const size_t *neighbours)
{
	size_t gID;
	size_t gSize;
	size_t nNeighbours;
	float3 normal;
	struct particle self;

	gID = get_global_id(0);
	gSize = get_global_size(0);
	nNeighbours = neighbours[gID * (MAX_NEIGHBOURS + 1)];
	normal = 0.0f;

	self = particles[gID];

	size_t j;
	struct particle other;
	for (size_t i = 0; i < nNeighbours; i++)
	{
		j = neighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		other = particles[j];
		
		normal += other.mass * weightMonaghanSplinePrime(self.pos - other.pos, self.interactionRadius) * fast_normalize(self.pos - other.pos) 
			      / other.density;
	}

	normal *= self.interactionRadius * self.mass * self.monaghanSplinePrimeNormalisation;

	particles[gID].normal = normal;
}

__kernel void nonPressureForces(__global struct particle *particles,
					   			__global __read_only const size_t *neighbours,
						  	    const struct fluidParams params)
{
	size_t gID;
	size_t gSize;	
	size_t nNeighbours;
	float3 viscosityForce;
	float3 cohesionForce;
	float3 curvatureForce;
	struct particle self;

	gID = get_global_id(0);
	gSize = get_global_size(0);
	viscosityForce = 0.0f;
	cohesionForce = 0.0f;
	curvatureForce = 0.0f;

	nNeighbours = neighbours[gID * (MAX_NEIGHBOURS + 1)];
	
	self = particles[gID];

	size_t j;
	float surfaceTensionFactor;
	float3 r;
	float3 direction;
	struct particle other;
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
			   			* direction * other.mass;
		
		/* Surface tension cohesion */
		cohesionForce += surfaceTensionFactor * other.mass * weightSurfaceTension(r, self.interactionRadius, self.surfaceTensionTerm) * direction;

		/* Surface tension curvature */
		curvatureForce += surfaceTensionFactor * (self.normal - other.normal);

	}	

	viscosityForce *= -self.monaghanSplinePrimeNormalisation * self.viscosity;
	cohesionForce *= -self.surfaceTensionNormalisation * self.surfaceTension * self.mass;
	curvatureForce *= -self.surfaceTension * self.mass;

	particles[gID].velocityAdvection = self.vel + (viscosityForce + cohesionForce + curvatureForce + params.gravity * self.mass) * params.timeStep / self.mass;
}

__kernel void initPressure(__global struct particle *particles,
						   __global __read_only const size_t *neighbours,
						   const struct fluidParams params)
{
	size_t gID;
	size_t nNeighbours;

	float density;
	float aii;
	float3 dii;

	struct particle self;
	
	gID = get_global_id(0);
	nNeighbours = neighbours[gID * (MAX_NEIGHBOURS + 1)];

	density = 0.0f;
	aii = 0.0f;
	dii = 0.0f;

	self = particles[gID];

	size_t j;
	float weight;
	float3 direction;
	struct particle other;

	/* Calculate density & dii */
	for (size_t i = 0; i < nNeighbours; i++)
	{
		j = neighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		other = particles[j];
		weight = weightMonaghanSplinePrime(self.pos - other.pos, self.interactionRadius);
		direction = normalize(self.pos - other.pos);

		dii += other.mass * weight * direction;
		density += other.mass * dot(self.velocityAdvection - other.velocityAdvection, weight * direction);
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
	if (isgreater(self.pos.x, 1.2f))
	{
		self.pos.x = 2.4f - self.pos.x;
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
	if (isgreater(self.pos.z, 1.8f))
	{
		self.pos.z = 3.6f - self.pos.z;
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
		
	self.pos = clamp(self.pos, -10.0f, 10.0f);

	particles[gID] = self;
}

__kernel void predictDensityPressure(__global struct particle *particles,
									 __global __write_only float *densityErrors,
									 __global __write_only float *pressureTemp, 
					   				 __global __read_only const size_t *neighbours,
	 								 const struct fluidParams params)
{
	size_t gID;
	size_t nNeighbours;
	float factor;
	struct particle self;

	gID = get_global_id(0);
	nNeighbours = neighbours[gID * (MAX_NEIGHBOURS + 1)];
	factor = 0.0f;
	self = particles[gID];
	
	size_t j;
	float weight;
	float3 direction;
	struct particle other;
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

	factor *= self.monaghanSplinePrimeNormalisation;	

	float temp;
	temp = 0.0f;

	/* Update pressure */
	if(self.advection != 0.0f)
		temp = (1.0f - params.relaxationFactor) * self.pressure + (1.0f * params.relaxationFactor / self.advection) * (self.restDensity - self.densityAdvection - factor);
	temp = max(temp, 0.0f);
	pressureTemp[gID] = temp;

	/* Predict density error */
//	particles[gID].density = self.densityAdvection + temp * self.advection + factor;
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
					   		 __global __read_only const size_t *neighbours,
							 __global __write_only float4 *pressure)
{
	size_t gID;
	size_t gSize;
	size_t nNeighbours;
	float pRhoSqr;
	float3 pressureForce;
	struct particle self;
	
	gID = get_global_id(0);
	gSize = get_global_size(0);

	nNeighbours = neighbours[gID * (MAX_NEIGHBOURS + 1)];
	self = particles[gID];

	pRhoSqr = self.pressure / pown(self.density, 2);
	pressureForce = 0.0f;

	
	size_t j;
	struct particle other;
	for (size_t i = 0; i < nNeighbours; i++)
	{
		j = neighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		other = particles[j];
		pressureForce += other.mass * mad(other.pressure, pown(other.density, -2), pRhoSqr) * weightMonaghanSplinePrime(self.pos - other.pos, self.interactionRadius) * fast_normalize(self.pos - other.pos);
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

