#include "cl.h"
#include "hash.h"
#include "log.h"
#include "particle.h"
#include "sph.h"

#include <stdio.h>
#include <time.h>

#define MAX_NEIGHBOURS 400
#define PROGRAM_FILE "src/sph.cl"

/* Find a GPU or CPU associated with the first available platform */
static cl_device_id create_device(void);

/* Create program from a file and compile it */
static cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename);

static bool checkDensityErrors(cl_float *densityError, size_t num_fluid_particles);

static uint64_t time_diff(const struct timespec * const cur,
		const struct timespec * const old);

void cl_initialise(struct sph *sph)
{
	cl_int err;

	sph->cl.density_error = calloc(sph->num_fluid_particles, sizeof(*sph->cl.density_error));

	sph->cl.device = create_device();
	sph->cl.context = clCreateContext(NULL, 1, &sph->cl.device, NULL, NULL, &err);

	if(err < 0) {
		log_error("Couldn't create a context\n");
		exit(1);   
	}

	/* Create data buffers */
	sph->cl.buffers.particle = clCreateBuffer(
			sph->cl.context,
			CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
			sizeof(*sph->particles) * sph->num_fluid_particles,
			sph->particles,
			&err);
	if(err < 0) {
		log_error("Couldn't create particle buffer\n");
		exit(1);   
	};

	sph->cl.buffers.boundary = clCreateBuffer(
			sph->cl.context,
			CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
			1,//sizeof(*sph->particles) * sph->num_boundary_particles,
			sph->particles,
			&err);
	if(err < 0) {
		log_error("Couldn't create boundary particle buffer\n");
		exit(1);   
	};

	sph->cl.buffers.density_error = clCreateBuffer(
			sph->cl.context,
			CL_MEM_WRITE_ONLY,
			sizeof(cl_float) * sph->num_fluid_particles,
			NULL,
			&err);
	if(err < 0) {
		log_error("Couldn't create density error buffer\n");
		exit(1);   
	};

	sph->cl.buffers.pressure = clCreateBuffer(
			sph->cl.context,
			CL_MEM_READ_WRITE,
			sizeof(cl_float4) * sph->num_fluid_particles,
			NULL,
			&err);
	if(err < 0) {
		log_error("Couldn't create pressure buffer\n");
		exit(1);   
	};

	sph->cl.buffers.pressure_temp = clCreateBuffer(
			sph->cl.context,
			CL_MEM_READ_WRITE,
			sizeof(cl_float) * sph->num_fluid_particles,
			NULL,
			&err);
	if(err < 0) {
		log_error("Couldn't create pressure temp buffer\n");
		exit(1);   
	};

	sph->cl.buffers.neighbour = clCreateBuffer(
			sph->cl.context,
			CL_MEM_READ_WRITE,
			sizeof(size_t) * sph->num_fluid_particles * (MAX_NEIGHBOURS + 1),
			NULL,
			&err);
	if(err < 0) {
		log_error("Couldn't create neighbour buffer\n");
		exit(1);   
	};

	sph->cl.buffers.boundary_neighbour = clCreateBuffer(
			sph->cl.context,
			CL_MEM_READ_WRITE,
			sizeof(size_t) * sph->num_fluid_particles * (MAX_NEIGHBOURS + 1),
			NULL,
			&err);
	if(err < 0) {
		log_error("Couldn't create boundary neighbour buffer\n");
		exit(1);   
	};

	sph->cl.buffers.hash = clCreateBuffer(
			sph->cl.context,
			CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			sizeof(struct hash) * 2 * sph->num_fluid_particles,
			sph->hash_map,
			&err);
	if(err < 0) {
		log_error("Couldn't create hash buffer\n");
		exit(1);   
	};

	//sph->cl.buffers.boundary_hash = clCreateBuffer(
	//		sph->cl.context,
	//		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	//		sizeof(hash_t) * boundary_hash_size,
	//		boundary_hash,
	//		&err);
	//if(err < 0) {
	//	log_error("Couldn't create boundary hash buffer\n");
	//	exit(1);   
	//};

	sph->cl.buffers.fluid = clCreateBuffer(sph->cl.context,
			CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			sizeof(sph->params),
			&sph->params,
			&err);
	if(err < 0) {
		log_error("Couldn't create parameter buffer\n");
		exit(1);   
	};

	/* Build the program and create the kernels */
	sph->cl.program = build_program(sph->cl.context, sph->cl.device, PROGRAM_FILE);

	sph->cl.kernels.non_pressure_forces = clCreateKernel(sph->cl.program, "non_pressure_forces", &err);
	if(err < 0) {
		log_error("Couldn't create non_pressure_forces kernel\n");
		exit(1);   
	};

	sph->cl.kernels.init_pressure = clCreateKernel(sph->cl.program, "init_pressure", &err);
	if(err < 0) {
		log_error("Couldn't create init_pressure kernel\n");
		exit(1);   
	};

	sph->cl.kernels.correct = clCreateKernel(sph->cl.program, "correct", &err);
	if(err < 0) {
		log_error("Couldn't create correct kernel\n");
		exit(1);   
	};

	sph->cl.kernels.update_density = clCreateKernel(sph->cl.program, "update_density", &err);
	if(err < 0) {
		log_error("Couldn't create update_density kernel\n");
		exit(1);   
	};

	sph->cl.kernels.update_normals = clCreateKernel(sph->cl.program, "update_normals", &err);
	if(err < 0) {
		log_error("Couldn't create update_normals kernel\n");
		exit(1);   
	};

	sph->cl.kernels.update_dij = clCreateKernel(sph->cl.program, "update_dij", &err);
	if(err < 0) {
		log_error("Couldn't create update_dij kernel\n");
		exit(1);   
	};

	sph->cl.kernels.update_position_velocity = clCreateKernel(sph->cl.program, "time_step", &err);
	if(err < 0) {
		log_error("Couldn't create update_position_velocity kernel\n");
		exit(1);   
	};

	sph->cl.kernels.update_boundary_volumes = clCreateKernel(sph->cl.program, "update_boundary_volumes", &err);
	if(err < 0) {
		log_error("Couldn't create update_boundary_volumes kernel\n");
		exit(1);   
	};

	sph->cl.kernels.predict_density_pressure = clCreateKernel(sph->cl.program, "predict_density_pressure", &err);
	if(err < 0) {
		log_error("Couldn't create predict_density_pressure kernel\n");
		exit(1);   
	};

	sph->cl.kernels.copy_pressure = clCreateKernel(sph->cl.program, "copy_pressure", &err);
	if(err < 0) {
		log_error("Couldn't create copy_pressure kernel\n");
		exit(1);   
	};

	sph->cl.kernels.pressure_forces = clCreateKernel(sph->cl.program, "pressure_forces", &err);
	if(err < 0) {
		log_error("Couldn't create pressure_forces kernel\n");
		exit(1);   
	};

	sph->cl.kernels.find_neighbours = clCreateKernel(sph->cl.program, "find_neighbours", &err);
	if(err < 0) {
		log_error("Couldn't create find_neighbours kernel\n");
		exit(1);   
	};

	/* Set kernel args */
	err  = clSetKernelArg (sph->cl.kernels.non_pressure_forces, 0, sizeof(cl_mem), &sph->cl.buffers.particle);
	err |= clSetKernelArg (sph->cl.kernels.non_pressure_forces, 1, sizeof(cl_mem), &sph->cl.buffers.boundary);
	err |= clSetKernelArg (sph->cl.kernels.non_pressure_forces, 2, sizeof(cl_mem), &sph->cl.buffers.neighbour);
	err |= clSetKernelArg (sph->cl.kernels.non_pressure_forces, 3, sizeof(cl_mem), &sph->cl.buffers.boundary_neighbour);
	err |= clSetKernelArg (sph->cl.kernels.non_pressure_forces, 4, sizeof(cl_mem), &sph->cl.buffers.fluid);

	if(err < 0) {
		log_error("Couldn't set non_pressure_forces kernel args\n");
		exit(1);   
	};


	err  = clSetKernelArg (sph->cl.kernels.init_pressure, 0, sizeof(cl_mem), &sph->cl.buffers.particle);
	err |= clSetKernelArg (sph->cl.kernels.init_pressure, 1, sizeof(cl_mem), &sph->cl.buffers.boundary);
	err |= clSetKernelArg (sph->cl.kernels.init_pressure, 2, sizeof(cl_mem), &sph->cl.buffers.neighbour);
	err |= clSetKernelArg (sph->cl.kernels.init_pressure, 3, sizeof(cl_mem), &sph->cl.buffers.boundary_neighbour);
	err |= clSetKernelArg (sph->cl.kernels.init_pressure, 4, sizeof(cl_mem), &sph->cl.buffers.fluid);

	if(err < 0) {
		log_error("Couldn't set init_pressure kernel args\n");
		exit(1);   
	};


	err  = clSetKernelArg (sph->cl.kernels.correct, 0, sizeof(cl_mem), &sph->cl.buffers.particle);
	err |= clSetKernelArg (sph->cl.kernels.correct, 1, sizeof(cl_mem), &sph->cl.buffers.boundary);
	err |= clSetKernelArg (sph->cl.kernels.correct, 2, sizeof(cl_mem), &sph->cl.buffers.neighbour);
	err |= clSetKernelArg (sph->cl.kernels.correct, 3, sizeof(cl_mem), &sph->cl.buffers.boundary_neighbour);
	err |= clSetKernelArg (sph->cl.kernels.correct, 4, sizeof(cl_mem), &sph->cl.buffers.fluid);

	if(err < 0) {
		log_error("Couldn't set correct kernel args\n");
		exit(1);   
	};


	err  = clSetKernelArg (sph->cl.kernels.update_density, 0, sizeof(cl_mem), &sph->cl.buffers.particle);
	err |= clSetKernelArg (sph->cl.kernels.update_density, 1, sizeof(cl_mem), &sph->cl.buffers.boundary);
	err |= clSetKernelArg (sph->cl.kernels.update_density, 2, sizeof(cl_mem), &sph->cl.buffers.neighbour);
	err |= clSetKernelArg (sph->cl.kernels.update_density, 3, sizeof(cl_mem), &sph->cl.buffers.boundary_neighbour);
	err |= clSetKernelArg (sph->cl.kernels.update_density, 4, sizeof(cl_mem), &sph->cl.buffers.fluid);

	if(err < 0) {
		log_error("Couldn't set update_density kernel args\n");
		exit(1);   
	};


	err  = clSetKernelArg (sph->cl.kernels.update_normals, 0, sizeof(cl_mem), &sph->cl.buffers.particle);
	err |= clSetKernelArg (sph->cl.kernels.update_normals, 1, sizeof(cl_mem), &sph->cl.buffers.neighbour);
	err |= clSetKernelArg (sph->cl.kernels.update_normals, 2, sizeof(cl_mem), &sph->cl.buffers.fluid);

	if(err < 0) {
		log_error("Couldn't set update_normals kernel args\n");
		exit(1);   
	};


	err  = clSetKernelArg (sph->cl.kernels.update_dij, 0, sizeof(cl_mem), &sph->cl.buffers.particle);
	err |= clSetKernelArg (sph->cl.kernels.update_dij, 1, sizeof(cl_mem), &sph->cl.buffers.neighbour);
	err |= clSetKernelArg (sph->cl.kernels.update_dij, 2, sizeof(cl_mem), &sph->cl.buffers.fluid);

	if(err < 0) {
		log_error("Couldn't set update_dij kernel args\n");
		exit(1);   
	};


	err  = clSetKernelArg (sph->cl.kernels.update_position_velocity, 0, sizeof(cl_mem), &sph->cl.buffers.particle);
	err |= clSetKernelArg (sph->cl.kernels.update_position_velocity, 1, sizeof(cl_mem), &sph->cl.buffers.pressure);
	err |= clSetKernelArg (sph->cl.kernels.update_position_velocity, 2, sizeof(cl_mem), &sph->cl.buffers.fluid);

	if(err < 0) { 
		log_error("Couldn't set update_position_velocity kernel args\n");
		exit(1);   
	};


	err  = clSetKernelArg (sph->cl.kernels.update_boundary_volumes, 0, sizeof(cl_mem), &sph->cl.buffers.boundary);
	err |= clSetKernelArg (sph->cl.kernels.update_boundary_volumes, 1, sizeof(cl_mem), &sph->cl.buffers.fluid);

	if(err < 0) { 
		log_error("Couldn't set update_boundary_volumes kernel args\n");
		exit(1);   
	};


	err  = clSetKernelArg (sph->cl.kernels.predict_density_pressure, 0, sizeof(cl_mem), &sph->cl.buffers.particle);
	err |= clSetKernelArg (sph->cl.kernels.predict_density_pressure, 1, sizeof(cl_mem), &sph->cl.buffers.boundary);
	err |= clSetKernelArg (sph->cl.kernels.predict_density_pressure, 2, sizeof(cl_mem), &sph->cl.buffers.density_error);
	err |= clSetKernelArg (sph->cl.kernels.predict_density_pressure, 3, sizeof(cl_mem), &sph->cl.buffers.pressure_temp);
	err |= clSetKernelArg (sph->cl.kernels.predict_density_pressure, 4, sizeof(cl_mem), &sph->cl.buffers.neighbour);
	err |= clSetKernelArg (sph->cl.kernels.predict_density_pressure, 5, sizeof(cl_mem), &sph->cl.buffers.boundary_neighbour);
	err |= clSetKernelArg (sph->cl.kernels.predict_density_pressure, 6, sizeof(cl_mem), &sph->cl.buffers.fluid);

	if(err < 0) {
		log_error("Couldn't set predict_density_pressure kernel args\n");
		exit(1);   
	};


	err  = clSetKernelArg (sph->cl.kernels.copy_pressure, 0, sizeof(cl_mem), &sph->cl.buffers.pressure_temp);
	err |= clSetKernelArg (sph->cl.kernels.copy_pressure, 1, sizeof(cl_mem), &sph->cl.buffers.particle);
	err |= clSetKernelArg (sph->cl.kernels.copy_pressure, 2, sizeof(cl_mem), &sph->cl.buffers.pressure);

	if(err < 0) {
		log_error("Couldn't set copy_pressure kernel args\n");
		exit(1);   
	};


	err  = clSetKernelArg (sph->cl.kernels.pressure_forces, 0, sizeof(cl_mem), &sph->cl.buffers.particle);
	err |= clSetKernelArg (sph->cl.kernels.pressure_forces, 1, sizeof(cl_mem), &sph->cl.buffers.boundary);
	err |= clSetKernelArg (sph->cl.kernels.pressure_forces, 2, sizeof(cl_mem), &sph->cl.buffers.neighbour);
	err |= clSetKernelArg (sph->cl.kernels.pressure_forces, 3, sizeof(cl_mem), &sph->cl.buffers.boundary_neighbour);
	err |= clSetKernelArg (sph->cl.kernels.pressure_forces, 4, sizeof(cl_mem), &sph->cl.buffers.pressure);
	err |= clSetKernelArg (sph->cl.kernels.pressure_forces, 5, sizeof(cl_mem), &sph->cl.buffers.fluid);

	if(err < 0) {
		log_error("Couldn't set pressure_forces kernel args\n");
		exit(1);   
	};

	err  = clSetKernelArg (sph->cl.kernels.find_neighbours, 0, sizeof(cl_mem), &sph->cl.buffers.particle);
	err |= clSetKernelArg (sph->cl.kernels.find_neighbours, 1, sizeof(cl_mem), &sph->cl.buffers.boundary);
	err |= clSetKernelArg (sph->cl.kernels.find_neighbours, 2, sizeof(cl_mem), &sph->cl.buffers.neighbour);
	err |= clSetKernelArg (sph->cl.kernels.find_neighbours, 3, sizeof(cl_mem), &sph->cl.buffers.boundary_neighbour);
	//err |= clSetKernelArg (sph->cl.kernels.find_neighbours, 4, sizeof(size_t), &sph->num_boundary_particles);
	err |= clSetKernelArg (sph->cl.kernels.find_neighbours, 4, sizeof(cl_mem), &sph->cl.buffers.fluid);
	//err |= clSetKernelArg (sph->cl.kernels.find_neighbours, 5, sizeof(cl_mem), &sph->cl.buffers.hash);

	if(err < 0) {
		log_error("Couldn't set find_neighbours kernel args\n");
		exit(1);   
	};

	/* Create a command queue */
	sph->cl.queue = clCreateCommandQueue(sph->cl.context, sph->cl.device, 0, &err);
	if(err < 0) {
		log_error("Couldn't create a command queue\n");
		exit(1);   
	};   
}

void cl_destroy(struct sph *sph)
{
	/* Deallocate resources */
	/* Kernels */
	clReleaseKernel(sph->cl.kernels.non_pressure_forces);
	clReleaseKernel(sph->cl.kernels.init_pressure);
	clReleaseKernel(sph->cl.kernels.update_density);
	clReleaseKernel(sph->cl.kernels.update_normals);
	clReleaseKernel(sph->cl.kernels.update_dij);
	clReleaseKernel(sph->cl.kernels.update_position_velocity);
	clReleaseKernel(sph->cl.kernels.predict_density_pressure);
	clReleaseKernel(sph->cl.kernels.copy_pressure);
	clReleaseKernel(sph->cl.kernels.pressure_forces);
	clReleaseKernel(sph->cl.kernels.find_neighbours);
	clReleaseKernel(sph->cl.kernels.correct);

	/* Buffers */
	clReleaseMemObject(sph->cl.buffers.particle);
	clReleaseMemObject(sph->cl.buffers.boundary);
	clReleaseMemObject(sph->cl.buffers.density_error);
	clReleaseMemObject(sph->cl.buffers.pressure);
	clReleaseMemObject(sph->cl.buffers.pressure_temp);
	clReleaseMemObject(sph->cl.buffers.neighbour);
	clReleaseMemObject(sph->cl.buffers.boundary_neighbour);
	clReleaseMemObject(sph->cl.buffers.hash);
	clReleaseMemObject(sph->cl.buffers.boundary_hash);
	clReleaseMemObject(sph->cl.buffers.fluid);

	/* OpenCL structures */
	clReleaseCommandQueue(sph->cl.queue);
	clReleaseProgram(sph->cl.program);
	clReleaseContext(sph->cl.context);
}

void cl_update(struct sph *sph)
{
	cl_int err;
	struct timespec time_old;
	struct timespec time_cur;

	clock_gettime(CLOCK_REALTIME, &time_old);
	/*		if (i == 101 * output_step)
			{
			printf("Gravity change!\n");
			simulation_params.gravity = (cl_float3){{0.0f, 0.0f, -9.81f}};
			err = clSetKernelArg (non_pressure_forces, 4, sizeof(fluid_t), &simulation_params);
			if(err < 0) {
			perror("Couldn't change gravity");
			exit(1);   
			};
			}
			*/		
	/* Refill hash map */
	memset(sph->hash_map, 0, sizeof(*sph->hash_map) * 2 * sph->num_fluid_particles);
	for (size_t i = 0; i < sph->num_fluid_particles; i++) {
		OPENCL_UINT idx = hash(sph->particles[i].pos, sph->params.interaction_radius, 2 * sph->num_fluid_particles);
		if (sph->hash_map[idx].num_indices == MAX_HASH) {
			continue;
		};
		sph->hash_map[idx].indices[sph->hash_map[idx].num_indices] = i;
		sph->hash_map[idx].num_indices++;
	}
	clock_gettime(CLOCK_REALTIME, &time_cur);
	printf("Generating hash map: %lf seconds\n", time_diff(&time_cur, &time_old) / 1000000000.0);
	time_old = time_cur;

	/* Update hash buffer */
	err = clEnqueueWriteBuffer(
			sph->cl.queue,
			sph->cl.buffers.hash,
			CL_TRUE,
			0,
			sizeof(*sph->hash_map) * 2 * sph->num_fluid_particles,
			sph->hash_map,
			0,
			NULL,
			NULL);

	if(err < 0) {
		perror("Couldn't enqueue read");
		exit(1);   
	};
	clFinish(sph->cl.queue);
	clock_gettime(CLOCK_REALTIME, &time_cur);
	printf("Writing hash buffer: %lf seconds\n", time_diff(&time_cur, &time_old) / 1000000000.0);
	time_old = time_cur;

	/* Update neighbours */
	err = clEnqueueNDRangeKernel(
			sph->cl.queue,
			sph->cl.kernels.find_neighbours,
			1,
			NULL,
			&sph->num_fluid_particles,
			NULL,
			0,
			NULL,
			NULL);
	if(err < 0) {
		perror("Couldn't enqueue find_neighbours kernel");
		exit(1);   
	};   
	clFinish(sph->cl.queue);
	clock_gettime(CLOCK_REALTIME, &time_cur);
	printf("Finding neighbours: %lf seconds\n", time_diff(&time_cur, &time_old) / 1000000000.0);
	time_old = time_cur;

	/* Correct kernel and density every density_correct_step steps */
	if (false)//(i % density_correct_step) == 0 && i > 0)
	{
		err = clEnqueueNDRangeKernel(
				sph->cl.queue,
				sph->cl.kernels.correct,
				1,
				NULL,
				&sph->num_fluid_particles,
				NULL,
				0,
				NULL,
				NULL);
		if(err < 0) {
			perror("Couldn't enqueue correct kernel");
			exit(1);   
		}  	
	} else {
		/* Update density */
		err = clEnqueueNDRangeKernel(
				sph->cl.queue,
				sph->cl.kernels.update_density,
				1,
				NULL,
				&sph->num_fluid_particles,
				NULL,
				0,
				NULL,
				NULL);
		if(err < 0) {
			perror("Couldn't enqueue update_density kernel");
			exit(1);   
		}  
	}
	clFinish(sph->cl.queue);
	clock_gettime(CLOCK_REALTIME, &time_cur);
	printf("Updating density: %lf seconds\n", time_diff(&time_cur, &time_old) / 1000000000.0);
	time_old = time_cur;
	/*****************************************/
	/*			err = clEnqueueReadBuffer(
				queue,
				particle_buffer,
				CL_TRUE,
				0,
				sizeof(struct particle) * num_fluid_particles,
				particles,
				0,
				NULL,
				NULL);

				if(err < 0) {
				perror("Couldn't enqueue read");
				exit(1);   
				};

				cl_float maxd = 0.0f;
				for (size_t j = 0; j < 101; j++)
				{
				if (particles[j].density > maxd)
				{
				maxd = particles[j].density;
				}
				}
				printf("Max: %f\n", maxd);
				exit(0);
				*/
	/****************************************/
	/* Update fluid normals */
	err = clEnqueueNDRangeKernel(
			sph->cl.queue,
			sph->cl.kernels.update_normals,
			1,
			NULL,
			&sph->num_fluid_particles,
			NULL,
			0,
			NULL,
			NULL);
	if(err < 0) {
		perror("Couldn't enqueue update_normals kernel");
		exit(1);   
	};   	
	clFinish(sph->cl.queue);
	clock_gettime(CLOCK_REALTIME, &time_cur);
	printf("Updating normals: %lf seconds\n", time_diff(&time_cur, &time_old) / 1000000000.0);
	time_old = time_cur;

	/* Calculate non-pressure forces */
	err = clEnqueueNDRangeKernel(
			sph->cl.queue,
			sph->cl.kernels.non_pressure_forces,
			1,
			NULL,
			&sph->num_fluid_particles,
			NULL,
			0,
			NULL,
			NULL);
	if(err < 0) {
		perror("Couldn't enqueue non_pressure_forces kernel");
		exit(1);   
	};   
	clFinish(sph->cl.queue);
	clock_gettime(CLOCK_REALTIME, &time_cur);
	printf("Non-pressure forces: %lf seconds\n", time_diff(&time_cur, &time_old) / 1000000000.0);
	time_old = time_cur;

	/* Predict density and pressure, and calculate IISPH coefficients */
	err = clEnqueueNDRangeKernel(
			sph->cl.queue,
			sph->cl.kernels.init_pressure,
			1,
			NULL,
			&sph->num_fluid_particles,
			NULL,
			0,
			NULL,
			NULL);
	if(err < 0) {
		perror("Couldn't enqueue init_pressure kernel");
		exit(1);   
	};   
	clFinish(sph->cl.queue);
	clock_gettime(CLOCK_REALTIME, &time_cur);
	printf("Initialising pressure: %lf seconds\n", time_diff(&time_cur, &time_old) / 1000000000.0);
	time_old = time_cur;

	/* Pressure loop */
	for(size_t l=0; l < 100; l++)
	{
		/* Calculate dij */
		err = clEnqueueNDRangeKernel(
				sph->cl.queue,
				sph->cl.kernels.update_dij,
				1,
				NULL,
				&sph->num_fluid_particles,
				NULL,
				0,
				NULL,
				NULL);
		if(err < 0) {
			printf("Iteration %zu\n", l);
			perror("Couldn't enqueue update_dij kernel");
			exit(1);   
		};   

		/* Predict density and its error and calculate pressure */
		err = clEnqueueNDRangeKernel(
				sph->cl.queue,
				sph->cl.kernels.predict_density_pressure,
				1,
				NULL,
				&sph->num_fluid_particles,
				NULL,
				0,
				NULL,
				NULL);
		if(err < 0) {
			printf("Iteration %zu\n", l);
			perror("Couldn't enqueue predict_density_pressure kernel");
			exit(1);   
		};   

		/* Update pressure */
		err = clEnqueueNDRangeKernel(
				sph->cl.queue,
				sph->cl.kernels.copy_pressure,
				1,
				NULL,
				&sph->num_fluid_particles,
				NULL,
				0,
				NULL,
				NULL);
		if(err < 0) {
			printf("Iteration %zu\n", l);
			perror("Couldn't enqueue copy_pressure kernel");
			exit(1);   
		};   

		if (l>1)
		{
			/* Read density error buffer */
			err = clEnqueueReadBuffer(
					sph->cl.queue,
					sph->cl.buffers.density_error,
					CL_TRUE,
					0,
					sizeof(cl_float) * sph->num_fluid_particles,
					sph->cl.density_error,
					0,
					NULL,
					NULL);

			if(err < 0) {
				perror("Couldn't enqueue read");
				exit(1);   
			};

			if (!checkDensityErrors(sph->cl.density_error, sph->num_fluid_particles))
			{
				break;
			}
		}
	}
	clFinish(sph->cl.queue);
	clock_gettime(CLOCK_REALTIME, &time_cur);
	printf("Pressure loop: %lf seconds\n", time_diff(&time_cur, &time_old) / 1000000000.0);
	time_old = time_cur;

	/* Compute pressure force */
	err = clEnqueueNDRangeKernel(
			sph->cl.queue,
			sph->cl.kernels.pressure_forces,
			1,
			NULL,
			&sph->num_fluid_particles,
			NULL,
			0,
			NULL,
			NULL);
	if(err < 0) {
		perror("Couldn't enqueue pressure_forces kernel");
		exit(1);   
	};   
	clFinish(sph->cl.queue);
	clock_gettime(CLOCK_REALTIME, &time_cur);
	printf("Pressure forces: %lf seconds\n", time_diff(&time_cur, &time_old) / 1000000000.0);
	time_old = time_cur;

	/* Update positions and velocities */
	err = clEnqueueNDRangeKernel(
			sph->cl.queue,
			sph->cl.kernels.update_position_velocity,
			1,
			NULL,
			&sph->num_fluid_particles,
			NULL,
			0,
			NULL,
			NULL);
	if(err < 0) {
		perror("Couldn't enqueue update_position_velocity kernel");
		exit(1);   
	};   
	clFinish(sph->cl.queue);
	clock_gettime(CLOCK_REALTIME, &time_cur);
	printf("Applying pressure: %lf seconds\n", time_diff(&time_cur, &time_old) / 1000000000.0);
	time_old = time_cur;

	/* Read particle array */
	err = clEnqueueReadBuffer(
			sph->cl.queue,
			sph->cl.buffers.particle,
			CL_TRUE,
			0,
			sizeof(*sph->particles) * sph->num_fluid_particles,
			sph->particles,
			0,
			NULL,
			NULL);

	if(err < 0) {
		perror("Couldn't enqueue read");
		exit(1);   
	};
	clFinish(sph->cl.queue);
	clock_gettime(CLOCK_REALTIME, &time_cur);
	printf("Reading data: %lf seconds\n", time_diff(&time_cur, &time_old) / 1000000000.0);
	time_old = time_cur;

	///* Re-hash particles */
	//hash_particles(particles, particle_hash, num_fluid_particles, particle_hash_size, simulation_params.grid_size);
	//
	//err = clEnqueueWriteBuffer(
	//		queue,
	//		hash_buffer,
	//		CL_TRUE,
	//		0,
	//		sizeof(hash_t) * particle_hash_size,
	//		particle_hash,
	//		0,
	//		NULL,
	//		NULL);
	//
	//if(err < 0) {
	//	perror("Couldn't enqueue read");
	//	exit(1);   
	//};
	//clFinish(sph->cl.queue);
	//printf("Re-hashing particles: %f seconds\n", (float)(clock() - t) / CLOCKS_PER_SEC);
	//t = clock();
}

/* Find a GPU or CPU associated with the first available platform */
cl_device_id create_device()
{
	cl_platform_id platform;
	cl_device_id dev;
	int err;

	char name[50];

	/* Identify a platform */
	err = clGetPlatformIDs(1, &platform, NULL);
	if(err < 0) {
		log_error("Couldn't identify a platform\n");
		exit(EXIT_FAILURE);
	} 

	clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(name), &name, NULL);
	log_info("Using platform %s\n", name);

	/* Access a device */
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
	if(err == CL_DEVICE_NOT_FOUND) {
		log_warning("No GPU found, reverting to CPU\n");
		err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
	}
	if(err < 0) {
		log_error("Couldn't access any devices\n");
		exit(EXIT_FAILURE);   
	}

	return dev;
}

/* Create program from a file and compile it */
cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename)
{
	cl_program program;
	FILE *program_handle;
	char *program_buffer, *program_log;
	size_t program_size, program_read_size, log_size;
	int err;

	/* Read program file and place content into buffer */
	program_handle = fopen(filename, "r");
	if(program_handle == NULL) {
		log_error("Couldn't find the program file\n");
		exit(EXIT_FAILURE);
	}
	fseek(program_handle, 0, SEEK_END);
	program_size = ftell(program_handle);
	rewind(program_handle);
	program_buffer = (char*)malloc(program_size + 1);
	program_buffer[program_size] = '\0';
	program_read_size = fread(program_buffer, sizeof(char), program_size, program_handle);
	if(program_read_size != program_size)
	{
		log_error("Error during program file read\n");
	}
	fclose(program_handle);

	/* Create program from file */
	program = clCreateProgramWithSource(ctx, 1, 
			(const char**)&program_buffer, &program_size, &err);
	if(err < 0) {
		log_error("Couldn't create the program\n");
		exit(EXIT_FAILURE);
	}
	free(program_buffer);

	/* Build program */
	err = clBuildProgram(program, 0, NULL, "-I.", NULL, NULL);
	if(err < 0) {

		/* Find size of log and print to std output */
		clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 
				0, NULL, &log_size);
		program_log = (char*) malloc(log_size + 1);
		program_log[log_size] = '\0';
		clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 
				log_size + 1, program_log, NULL);
		log_error("%s\n", program_log);
		free(program_log);
		exit(EXIT_FAILURE);
	}

	return program;
}

bool checkDensityErrors(cl_float *densityError, size_t num_fluid_particles)
{
	//	float max = 0.0f;
	//	size_t index = 0;
	size_t num = 0;
	cl_float sum = 0.0f;
	for (size_t i = 0; i < num_fluid_particles; i++)
	{
		//		if (densityError[i] > max){
		//			max = densityError[i];
		//			index = i;}
		if (densityError[i] > 0.0f)
		{
			sum += densityError[i];
			num++;
		}
	}

	//	printf("Max: %lu,   %.4f\n", index, max);
	//	printf("Average:   %.4f\n", sum / (cl_float)num);

	if (num > 0 && sum / (cl_float)num > 0.001f)
		return true;

	return false;
}

uint64_t time_diff(const struct timespec * const cur,
		const struct timespec * const old)
{
	uint64_t sec = (uint64_t)(cur->tv_sec - old->tv_sec);
	uint64_t nsec;
	if (sec == 0) {
		nsec = (uint64_t)(cur->tv_nsec - old->tv_nsec);
		return nsec;
	}
	nsec = 1000000000 * sec;
	if (old->tv_nsec > cur->tv_nsec) {
		nsec -= (uint64_t)(old->tv_nsec - cur->tv_nsec);
	} else {
		nsec += (uint64_t)(cur->tv_nsec - old->tv_nsec);
	}
	return nsec;
}
