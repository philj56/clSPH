#define OPENCL_COMPILING
#include "src/fluid_params.h"
#include "src/particle.h"

float weight_monaghan_spline(float3 p12, float h);

float weight_monaghan_spline_prime(float3 p12, float h);

float weight_surface_tension(float3 p12, float h, float extra_term);

float weight_adhesion(float3 p12, float h);

float viscosity_monaghan(float3 p12,
						float3 v12,
						float  rho_avg,
						float  interaction_radius);

/* Simple neighbour search */
__kernel void find_neighbours (__global const struct particle *particles,
							  __global const struct particle *boundary_particles,
					   		  __global size_t *neighbours,
					   		  __global size_t *boundary_neighbours,
							  //const size_t n_boundary_particles,
					   		  __constant struct fluid_params *params)
{
	size_t gID = get_global_id(0);
	size_t g_size = get_global_size(0);
	size_t n_neighbours = 0;
	size_t n_boundary_neighbours = 0;
	float dist;
	struct particle self = particles[gID];

	for (size_t i = 0; i < g_size && n_neighbours < MAX_NEIGHBOURS; i++)
	{
		dist = distance(self.pos, particles[i].pos);
		if (isless(dist, params[self.fluid_index].interaction_radius) && i != gID)
		{
			n_neighbours++;
			neighbours[gID * (MAX_NEIGHBOURS + 1) + n_neighbours] = i;
		}
	}
	
	//for (size_t i = 0; i < n_boundary_particles && n_boundary_neighbours < MAX_NEIGHBOURS; i++)
	//{
	//	dist = distance(self.pos, boundary_particles[i].pos);
	//	if (isless(dist, params[self.fluid_index].interaction_radius))
	//	{
	//		n_boundary_neighbours++;
	//		boundary_neighbours[gID * (MAX_NEIGHBOURS + 1) + n_boundary_neighbours] = i;
	//	}
	//}

	neighbours[gID * (MAX_NEIGHBOURS + 1)] = n_neighbours;
	boundary_neighbours[gID * (MAX_NEIGHBOURS + 1)] = n_boundary_neighbours;
}

__kernel void update_boundary_volumes(__global struct particle *particles,
									__constant struct fluid_params *params)
{
	size_t gID = get_global_id(0);
	size_t g_size = get_global_size(0);
	float sum = 0.0f;
	struct particle self = particles[gID];

	for (size_t i = 0; i < g_size; i++)
	{
		if (distance(self.pos, particles[i].pos) <= params[self.fluid_index].interaction_radius)
			sum += weight_monaghan_spline(self.pos - particles[i].pos, params[self.fluid_index].interaction_radius);
	}

	particles[gID].volume = 1.0f / (sum * params[self.fluid_index].monaghan_spline_normalisation);
}

__kernel void update_density(__global struct particle *particles,
							__global const struct particle *boundary_particles,
					   		__global const size_t *neighbours,
					   		__global const size_t *boundary_neighbours,
							__constant struct fluid_params *params)
{
	size_t gID = get_global_id(0);
	size_t n_neighbours = neighbours[gID * (MAX_NEIGHBOURS + 1)];
	size_t n_boundary_neighbours = boundary_neighbours[gID * (MAX_NEIGHBOURS + 1)];
	float density = 1.0f;
	struct particle self = particles[gID];
	
	size_t j;
	struct particle other;

	/* Fluid particles */
	for (size_t i = 0; i < n_neighbours; i++)
	{
		j = neighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		other = particles[j];

		density += weight_monaghan_spline(self.pos - other.pos, params[self.fluid_index].interaction_radius);
	}

	density *= params[self.fluid_index].mass;
	
	/* Boundary particles */
	for (size_t i = 0; i < n_boundary_neighbours; i++)
	{
		j = boundary_neighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		other = boundary_particles[j];

		density += params[self.fluid_index].rest_density * other.volume * weight_monaghan_spline(self.pos - other.pos, params[self.fluid_index].interaction_radius);
	}

	density *= params[self.fluid_index].monaghan_spline_normalisation;// * self.kernel_correction;

	particles[gID].density = density;	
}

__kernel void correct(__global struct particle *particles,
							__global const struct particle *boundary_particles,
					   		__global const size_t *neighbours,
					   		__global const size_t *boundary_neighbours,
							__constant struct fluid_params *params)
{
	size_t gID = get_global_id(0);
	size_t n_neighbours = neighbours[gID * (MAX_NEIGHBOURS + 1)];
	size_t n_boundary_neighbours = boundary_neighbours[gID * (MAX_NEIGHBOURS + 1)];
	float density = 1.0f;
	float correction = 0.0f;
	struct particle self = particles[gID];
	
	size_t j;
	struct particle other;
	
	/* Fluid particles */
	for (size_t i = 0; i < n_neighbours; i++)
	{
		j = neighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		other = particles[j];

		correction += (self.fluid_index == other.fluid_index) * params[self.fluid_index].mass * weight_monaghan_spline(self.pos - other.pos, params[self.fluid_index].interaction_radius) / other.density;
		density += params[self.fluid_index].mass * weight_monaghan_spline(self.pos - other.pos, params[self.fluid_index].interaction_radius);
	}

	correction += 1.0f / self.density;

	correction *= params[self.fluid_index].monaghan_spline_normalisation;
	density *= params[self.fluid_index].monaghan_spline_normalisation;
	
	/* Boundary particles */
	for (size_t i = 0; i < n_boundary_neighbours; i++)
	{
		j = boundary_neighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		other = boundary_particles[j];

		correction += other.volume * weight_monaghan_spline(self.pos - other.pos, params[self.fluid_index].interaction_radius);
		density += params[self.fluid_index].rest_density * other.volume * weight_monaghan_spline(self.pos - other.pos, params[self.fluid_index].interaction_radius);
	}


	particles[gID].density = 1.0f / correction * density;

	/* Skip correction if there are less than 2 neighbours or more than 2 boundary neighbours */
	if (n_neighbours <= 2 && n_boundary_neighbours <= 2)
	{
		particles[gID].density = density;
	}
//	particles[gID].kernel_correction = 1.0f / correction;*/
}

__kernel void update_normals(__global struct particle *particles,
							__global const size_t *neighbours,
							__constant struct fluid_params *params)
{
	size_t gID = get_global_id(0);
	size_t n_neighbours = neighbours[gID * (MAX_NEIGHBOURS + 1)];
	float3 normal = 0.0f;
	struct particle self = particles[gID];

	size_t j;
	struct particle other;

	for (size_t i = 0; i < n_neighbours; i++)
	{
		j = neighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		other = particles[j];
		
		normal += (self.fluid_index == other.fluid_index) * params[self.fluid_index].mass * weight_monaghan_spline_prime(self.pos - other.pos, params[self.fluid_index].interaction_radius) * fast_normalize(self.pos - other.pos) 
			      / other.density;
	}

	normal *= params[self.fluid_index].interaction_radius * params[self.fluid_index].mass * params[self.fluid_index].monaghan_spline_prime_normalisation;

	particles[gID].normal = normal;
}

__kernel void non_pressure_forces(__global struct particle *particles,
								__global const struct particle *boundary_particles,
					   			__global const size_t *neighbours,
					   			__global const size_t *boundary_neighbours,
						  	    __constant struct fluid_params *params)
{
	size_t gID = get_global_id(0);
	size_t n_neighbours = neighbours[gID * (MAX_NEIGHBOURS + 1)];
	size_t n_boundary_neighbours = boundary_neighbours[gID * (MAX_NEIGHBOURS + 1)];
	float3 viscosity_force = 0.0f;
	float3 cohesion_force = 0.0f;
	float3 curvature_force = 0.0f;
	float3 adhesion_force = 0.0f;
	struct particle self = particles[gID];

	size_t j;
	float surface_tension_factor;
	float3 r;
	float3 direction;
	struct particle other;

	/* Fluid particles */
	for (size_t i = 0; i < n_neighbours; i++)
	{
		j = neighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		other = particles[j];
		r = self.pos - other.pos;
		direction = normalize(r);

		surface_tension_factor = 2.0f * params[self.fluid_index].rest_density / (self.density + other.density);

		/* Viscosity */
		viscosity_force += viscosity_monaghan(r, 
								   self.vel - other.vel, 
								   (self.density + other.density) * 0.5f, 
								   params[self.fluid_index].interaction_radius)
			   			* weight_monaghan_spline_prime(r, params[self.fluid_index].interaction_radius)
			   			* direction * params[other.fluid_index].mass * params[self.fluid_index].viscosity;
		
		/* Surface tension cohesion */
		cohesion_force += (self.fluid_index == other.fluid_index) * surface_tension_factor * params[self.fluid_index].mass 
				       * weight_surface_tension(r, params[self.fluid_index].interaction_radius, params[self.fluid_index].surface_tension_term) * direction;

		/* Surface tension curvature */
		curvature_force += (self.fluid_index == other.fluid_index) * surface_tension_factor * (self.normal - other.normal);

	}	
	
	/* Boundary particles */
	for (size_t i = 0; i < n_boundary_neighbours; i++)
	{
		j = boundary_neighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		other = boundary_particles[j];
		r = self.pos - other.pos;
		direction = normalize(r);

		/* Viscosity */
		viscosity_force += viscosity_monaghan(r, 
								   self.vel, 
								   self.density * 2.0f, 
								   params[self.fluid_index].interaction_radius)
			   			* weight_monaghan_spline_prime(r, params[self.fluid_index].interaction_radius)
			   			* direction * params[self.fluid_index].rest_density * other.volume * params[other.fluid_index].viscosity;	

//		adhesion_force += self.rest_density * boundary.volume * weight_adhesion(r, self.interaction_radius) * direction * boundary.adhesion_modifier;
	}	


	viscosity_force *= -params[self.fluid_index].monaghan_spline_prime_normalisation * params[self.fluid_index].mass;
	cohesion_force *= -params[self.fluid_index].surface_tension_normalisation * params[self.fluid_index].surface_tension * params[self.fluid_index].mass;
	curvature_force *= -params[self.fluid_index].surface_tension * params[self.fluid_index].mass;
	adhesion_force *= -params[self.fluid_index].adhesion * params[self.fluid_index].mass;

	particles[gID].velocity_advection = self.vel + (viscosity_force + cohesion_force + curvature_force + adhesion_force + params[self.fluid_index].gravity * params[self.fluid_index].mass) * params[self.fluid_index].time_step / params[self.fluid_index].mass;
}

__kernel void init_pressure(__global struct particle *particles,
						   __global const struct particle *boundary_particles,
					   	   __global const size_t *neighbours,
					   	   __global const size_t *boundary_neighbours,
						   __constant struct fluid_params *params)
{
	size_t gID = get_global_id(0);
	size_t n_neighbours = neighbours[gID * (MAX_NEIGHBOURS + 1)];
	size_t n_boundary_neighbours = boundary_neighbours[gID * (MAX_NEIGHBOURS + 1)];
	float density = 0.0f;
	float aii = 0.0f;
	float3 dii = 0.0f;
	struct particle self = particles[gID];
	
	size_t j;
	float weight;
	float3 direction;
	float3 dji; 
	struct particle other;

	/* Calculate density & dii */
	for (size_t i = 0; i < n_neighbours; i++)
	{
		j = neighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		other = particles[j];
		weight = weight_monaghan_spline_prime(self.pos - other.pos, params[self.fluid_index].interaction_radius);
		direction = normalize(self.pos - other.pos);

		dii += params[other.fluid_index].mass * weight * direction;
		density += params[self.fluid_index].mass * dot(self.velocity_advection - other.velocity_advection, weight * direction);
	}
	
	/* Boundary particles */
	for (size_t i = 0; i < n_boundary_neighbours; i++)
	{
		j = boundary_neighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		other = boundary_particles[j];
		weight = weight_monaghan_spline_prime(self.pos - other.pos, params[self.fluid_index].interaction_radius);
		direction = normalize(self.pos - other.pos);

		dii += params[self.fluid_index].rest_density * other.volume * weight * direction;
		density += params[self.fluid_index].rest_density * other.volume * dot(self.velocity_advection, weight * direction);
	}
	
	dii *= -pown(params[self.fluid_index].time_step, 2) * pown(self.density, -2) * params[self.fluid_index].monaghan_spline_prime_normalisation;
	density *= params[self.fluid_index].time_step * params[self.fluid_index].monaghan_spline_prime_normalisation;
	
	/* Predict density */
	self.density_advection = self.density + density;
	
	/* Initialise pressure */
	self.pressure *= 0.5f;

	/* Calculate aii */
	for (size_t i = 0; i < n_neighbours; i++)
	{
		j = neighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		other = particles[j];
		weight = weight_monaghan_spline_prime(self.pos - other.pos, params[self.fluid_index].interaction_radius);
		direction = normalize(self.pos - other.pos);
		

		dji = params[self.fluid_index].mass * pown(params[self.fluid_index].time_step, 2) * pown(self.density, -2) * weight * direction * params[self.fluid_index].monaghan_spline_prime_normalisation;
		aii += params[other.fluid_index].mass * dot(dii - dji, weight * direction);
	}
	
	/* Boundary particles */
	for (size_t i = 0; i < n_boundary_neighbours; i++)
	{
		j = boundary_neighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		other = boundary_particles[j];
		weight = weight_monaghan_spline_prime(self.pos - other.pos, params[self.fluid_index].interaction_radius);
		direction = normalize(self.pos - other.pos);

		dji = params[self.fluid_index].rest_density * other.volume * pown(params[self.fluid_index].time_step, 2) * pown(self.density, -2) * weight * direction * params[self.fluid_index].monaghan_spline_prime_normalisation;
		aii += params[self.fluid_index].rest_density * other.volume * dot(dii - dji, weight * direction);
	}

	aii *= params[self.fluid_index].monaghan_spline_prime_normalisation;

	/* Update coefficients */
	self.displacement = dii;
	self.advection = aii;

	particles[gID] = self;
}

__kernel void update_dij(__global struct particle *particles,
						__global const size_t *neighbours,
						__constant struct fluid_params *params)
{
	size_t gID = get_global_id(0);
	size_t n_neighbours = neighbours[gID * (MAX_NEIGHBOURS + 1)];
	float3 sum = 0.0f;
	struct particle self = particles[gID];

	size_t j;
	struct particle other;

	for (size_t i = 0; i < n_neighbours; i++)
	{
		j = neighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		other = particles[j];

		sum += params[other.fluid_index].mass * other.pressure * pown(other.density, -2) * weight_monaghan_spline_prime(self.pos - other.pos, params[self.fluid_index].interaction_radius) * normalize(self.pos - other.pos);
	}

	sum *= -pown(params[self.fluid_index].time_step, 2) * params[self.fluid_index].monaghan_spline_prime_normalisation;

	particles[gID].sum_pressure_movement = sum;
}

__kernel void time_step(__global struct particle *particles,
					   __global const float4 *force,
					   __constant struct fluid_params *params)
{
	size_t gID = get_global_id(0);
	struct particle self = particles[gID];

	self.vel = self.velocity_advection + params[self.fluid_index].time_step * force[gID].xyz / params[self.fluid_index].mass;
	self.pos += self.vel * params[self.fluid_index].time_step;

	/* For now, clamp particle position to a range */
	particles[gID].pos = clamp(self.pos, -10.0f, 10.0f);
	particles[gID].vel = self.vel;
}

__kernel void predict_density_pressure(__global struct particle *particles,
									 __global const struct particle *boundary_particles,
									 __global float *density_errors,
									 __global float *pressure_temp, 
					   				 __global const size_t *neighbours,
					   				 __global const size_t *boundary_neighbours,
	 								 __constant struct fluid_params *params)
{
	size_t gID = get_global_id(0);
	size_t n_neighbours = neighbours[gID * (MAX_NEIGHBOURS + 1)];
	size_t n_boundary_neighbours = boundary_neighbours[gID * (MAX_NEIGHBOURS + 1)];
	float factor = 0.0f;
	float temp = 0.0f;
	struct particle self = particles[gID];
	
	size_t j;
	float weight;
	float3 direction;
	struct particle other;

	/* Fluid particles */
	for (size_t i = 0; i < n_neighbours; i++)
	{
		j = neighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		other = particles[j];
		weight = weight_monaghan_spline_prime(self.pos - other.pos, params[self.fluid_index].interaction_radius);
		direction = normalize(self.pos - other.pos);

		factor += params[other.fluid_index].mass * dot(self.sum_pressure_movement - other.displacement * other.pressure - other.sum_pressure_movement 
		       + params[self.fluid_index].mass * pown(params[self.fluid_index].time_step, 2) * pown(self.density, -2) * weight * params[self.fluid_index].monaghan_spline_prime_normalisation 
			   * direction * self.pressure, weight*direction);
	}

	/* Boundary particles */
	for (size_t i = 0; i < n_boundary_neighbours; i++)
	{
		j = boundary_neighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		other = boundary_particles[j];
		weight = weight_monaghan_spline_prime(self.pos - other.pos, params[self.fluid_index].interaction_radius);
		direction = normalize(self.pos - other.pos);

		factor += params[self.fluid_index].rest_density * other.volume * dot(self.sum_pressure_movement, weight*direction);
	}

	factor *= params[self.fluid_index].monaghan_spline_prime_normalisation;	

	/* Update pressure */
	if(self.advection != 0.0f)
		temp = (1.0f - params[self.fluid_index].relaxation_factor) * self.pressure + (1.0f * params[self.fluid_index].relaxation_factor / self.advection) * (params[self.fluid_index].rest_density - self.density_advection - factor);
//	temp = max(temp, 0.0f);
	pressure_temp[gID] = max(temp, 0.0f);

/*	if (gID == 680)
	{
		printf("Advection:    %f\n", self.advection);
		printf("Density:      %f\n", self.density);
		printf("Density_advec: %f\n", self.density_advection);
		printf("Pressure:     %f\n", temp);
	}
*/
	/* Predict density error */
//	particles[gID].density_advection = self.density_advection + temp * self.advection + factor;
	density_errors[gID] = (self.density_advection + temp * self.advection + factor - params[self.fluid_index].rest_density) / params[self.fluid_index].rest_density;
}

__kernel void copy_pressure(__global const float *in,
						   __global struct particle *particles,
						   __global float4 *out)
{
	size_t gID = get_global_id(0);

	out[gID].w = in[gID];
	particles[gID].pressure = in[gID];
}

__kernel void pressure_forces(__global const struct particle *particles,
					   		 __global const struct particle *boundary_particles,
					   		 __global const size_t *neighbours,
					   		 __global const size_t *boundary_neighbours,
							 __global float4 *pressure,
							 __constant struct fluid_params *params)
{
	size_t gID = get_global_id(0);
	size_t n_neighbours = neighbours[gID * (MAX_NEIGHBOURS + 1)];
	size_t n_boundary_neighbours = boundary_neighbours[gID * (MAX_NEIGHBOURS + 1)];
	float3 pressure_force = 0.0f;
	struct particle self = particles[gID];
	float p_rho_sqr = self.pressure / pown(self.density, 2);
	
	size_t j;
	struct particle other;
	
	/* Fluid particles */
	for (size_t i = 0; i < n_neighbours; i++)
	{
		j = neighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		other = particles[j];
//		pressure_force += other.mass * mad(other.pressure, pown(other.density, -2), p_rho_sqr) * weight_monaghan_spline_prime(self.pos - other.pos, self.interaction_radius) * fast_normalize(self.pos - other.pos);
		pressure_force += params[other.fluid_index].mass * other.pressure / (self.density * other.density) * weight_monaghan_spline_prime(self.pos - other.pos, params[self.fluid_index].interaction_radius) * fast_normalize(self.pos - other.pos);
	}
	
	/* Boundary particles */
	for (size_t i = 0; i < n_boundary_neighbours; i++)
	{
		j = boundary_neighbours[gID * (MAX_NEIGHBOURS + 1) + 1 + i];
		other = boundary_particles[j];
		pressure_force += other.volume * params[self.fluid_index].rest_density * p_rho_sqr * weight_monaghan_spline_prime(self.pos - other.pos, params[self.fluid_index].interaction_radius) * fast_normalize(self.pos - other.pos);
	}

	pressure_force *= -params[self.fluid_index].monaghan_spline_prime_normalisation * params[self.fluid_index].mass;

	pressure[gID].xyz = pressure_force;
	
	if(gID == 925 && (self.pos.x > 9.0f || self.pos.x < -9.0f))
		printf("%u:\npos:\t%.16v3hlf\nvel:\t%.16v3hlf\ndensity:\t%.16f\npressure:\t%.16f\nadvec:\t%.16f\ndensity_advec:\t%.16f\n", gID, 
				self.pos, self.vel, self.density, self.pressure, self.advection, self.density_advection);
}

float weight_monaghan_spline(float3 p12, float h)
{
	float q = fast_length(p12) / (0.5f * h);

	return islessequal(q, 1.0f) * mad(q * q, mad(q, 0.75f, -1.5f), 1.0f) + 			/* q < 1 */
		   isgreater(q, 1.0f) * islessequal(q, 2.0f) * 0.25f * pow(2.0f - q, 3.0f);	/* 1 < q < 2 */
}

float weight_monaghan_spline_prime(float3 p12, float h)
{
	float r = fast_length(p12);
	h *= 0.5f;

	return 0.75f * pown(h, -2) * (islessequal(r, h) * r * mad(r, 3.0f, -4.0f * h)								/* r < 0.5h */
								  - isgreater(r, h) * islessequal(r, 2.0f * h) * pown(mad(-2.0f, h, r), 2));	/* 0.5h < r < h */
}

float weight_surface_tension(float3 p12, float h, float extra_term)
{
	float r = fast_length(p12);

	return isgreater(r, 0.001f * h) * islessequal(r, h) * pown((h - r) * r, 3) 			/* r < h */
		   * (1.0f + islessequal(2.0f * r, h)) + islessequal(2.0f * r, h) * extra_term;	/* r < h/2 */
}

float weight_adhesion(float3 p12, float h)
{
	float r = fast_length(p12);

	return sqrt(sqrt(isgreaterequal(2.0f * r, h) * islessequal(r, h) * (- 4.0f * r * r / h + 6.0f * r - 2.0f * h )));	/* h/2 < r < h */
}


// v12:     v1 - v2
// p12:     p1 - p2
// rho_avg:  average of (rho1, rho2)
float viscosity_monaghan(float3 p12,
						float3 v12,
						float  rho_avg,
						float  interaction_radius)
{
	float vd = dot(v12, p12);
	
	// Arbitrary terms - see Monaghan 1992
	float alpha = 1;
	float beta = 2;
	float eta_sqr = 0.01 * interaction_radius * interaction_radius;

	// Take c to be 1400 m/s for now
	float c = 1400;

	float mu12  = (interaction_radius * vd) / (dot(p12, p12) + eta_sqr);

	return isless(vd, 0.0f) * mu12 * mad(-alpha, c, beta * mu12) / rho_avg;	
}

