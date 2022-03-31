#ifndef CLSPH_CL_H
#define CLSPH_CL_H

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

struct sph;

struct clsph_cl {
	/* OpenCL data structures */
	cl_device_id device;
	cl_context context;
	cl_command_queue queue;
	cl_program program;

	cl_float *density_error;
	
	/* Kernels */
	struct {
		cl_kernel non_pressure_forces;
		cl_kernel init_pressure;
		cl_kernel correct;
		cl_kernel update_density;
		cl_kernel update_normals;
		cl_kernel update_dij;
		cl_kernel update_position_velocity;
		cl_kernel update_boundary_volumes;
		cl_kernel predict_density_pressure;
		cl_kernel copy_pressure;
		cl_kernel pressure_forces;
		cl_kernel find_neighbours;
	} kernels;

	/* Buffers */
	struct {
		cl_mem particle;
		cl_mem boundary;
		cl_mem density_error;
		cl_mem pressure;
		cl_mem pressure_temp;
		cl_mem neighbour;
		cl_mem boundary_neighbour;
		cl_mem hash;
		cl_mem boundary_hash;
		cl_mem fluid;
	} buffers;
};

void cl_initialise(struct sph *sph);
void cl_destroy(struct sph *sph);
void cl_update(struct sph *sph);

#endif /* CLSPH_CL_H */
