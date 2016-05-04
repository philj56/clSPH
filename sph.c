#define _CRT_SECURE_NO_WARNINGS
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#define PROGRAM_FILE "sph.cl"

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <sys/types.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "fluidParams.h"

/* Find a GPU or CPU associated with the first available platform */
cl_device_id create_device() {

	cl_platform_id platform;
	cl_device_id dev;
	int err;

	/* Identify a platform */
	err = clGetPlatformIDs(1, &platform, NULL);
	if(err < 0) {
		perror("Couldn't identify a platform");
		exit(1);
	} 

	/* Access a device */
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
	if(err == CL_DEVICE_NOT_FOUND) {
		err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
	}
	if(err < 0) {
		perror("Couldn't access any devices");
		exit(1);   
	}

	return dev;
}

/* Create program from a file and compile it */
cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename) {

	cl_program program;
	FILE *program_handle;
	char *program_buffer, *program_log;
	size_t program_size, program_read_size, log_size;
	int err;

	/* Read program file and place content into buffer */
	program_handle = fopen(filename, "r");
	if(program_handle == NULL) {
		perror("Couldn't find the program file");
		exit(1);
	}
	fseek(program_handle, 0, SEEK_END);
	program_size = ftell(program_handle);
	rewind(program_handle);
	program_buffer = (char*)malloc(program_size + 1);
	program_buffer[program_size] = '\0';
	program_read_size = fread(program_buffer, sizeof(char), program_size, program_handle);
	if(program_read_size != program_size)
	{
		perror("Error during program file read");
	}
	fclose(program_handle);

	/* Create program from file */
	program = clCreateProgramWithSource(ctx, 1, 
		(const char**)&program_buffer, &program_size, &err);
	if(err < 0) {
		perror("Couldn't create the program");
		exit(1);
	}
	free(program_buffer);

	/* Build program */
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if(err < 0) {

		/* Find size of log and print to std output */
		clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 
				0, NULL, &log_size);
		program_log = (char*) malloc(log_size + 1);
		program_log[log_size] = '\0';
		clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 
				log_size + 1, program_log, NULL);
		printf("%s\n", program_log);
		free(program_log);
		exit(1);
	}

	return program;
}

// Quick and dirty random float in range
float randRange (float min, float max)
{
	return random() * (max - min) / RAND_MAX + min;
}

bool checkDensityErrors(cl_float *densityError, size_t num_particles)
{
	cl_float sum = 0.0f;
	for (size_t i = 0; i < num_particles; i++)
	{
		if (densityError[i] > 0.0f)
			sum += densityError[i];
	}
	
	if (sum / (cl_float)num_particles > 1.0f)
		return true;

	return false;
}

int main() {

	/* OpenCL data structures */
	cl_device_id device;
	cl_context context;
	cl_command_queue queue;
	cl_program program;
	cl_int err;
	
	/* Kernels */
	cl_kernel nonPressureForces;
	cl_kernel initPressure;
	cl_kernel updateDensity;
	cl_kernel updateNormals;
	cl_kernel updateDij;
	cl_kernel updateVelocity;
	cl_kernel updatePositionVelocity;
	cl_kernel predictDensityPressure;
	cl_kernel pressureForces;
	cl_kernel findNeighbours;

	/* Data */
	const size_t resolution = 10;
	const size_t num_particles = resolution*resolution*resolution;
	const size_t work_units = num_particles;
	const size_t outputStep = 1;

	struct fluidParams simulationParams;
	
	cl_float3 pos[num_particles];
	cl_float3 vel[num_particles];
	cl_float densityError[num_particles];

	/* Buffers */
	cl_mem posBuffer;
	cl_mem velBuffer;
	cl_mem densityBuffer;
	cl_mem densityGuessBuffer;
	cl_mem densityErrorBuffer;
	cl_mem normalBuffer;
	cl_mem nonPressureForcesBuffer;
	cl_mem pressureBuffer;
	cl_mem coefficientBuffer;
	cl_mem dijBuffer;
	cl_mem neighbourBuffer;

	srandom(time(NULL));

	/* Setup input data */	
	/* Use SI units */
	{
		const cl_float  restDensity = 1000.0f;
		const cl_float  particleMass = restDensity / num_particles;
		const cl_float  particleRadius = 1.0f/resolution;
		const cl_float  interactionRadius = 2.0f * particleRadius;
		const cl_float  timeStep = 0.004f;
		const cl_float  viscosity = 0.001f;
		const cl_float  surfaceTension = 0.0228f;
		const cl_float3 gravity = {{0.0f, 0.0f, 0.0f}};
		simulationParams = newFluidParams (particleMass, 
						   particleRadius, 
						   restDensity,
						   interactionRadius,
						   timeStep,
						   viscosity,
						   surfaceTension,
						   gravity);
	}

	for (size_t x = 0; x < resolution; x++)
	{
		for (size_t y = 0; y < resolution; y++)
		{
			for (size_t z = 0; z < resolution; z++)
			{
				pos[(x * resolution + y) * resolution + z] = (cl_float3) {{((float)x-resolution/2)*simulationParams.particleRadius,// + randRange(-0.01f, 0.01f),
											   ((float)y-resolution/2)*simulationParams.particleRadius,// + randRange(-0.01f, 0.01f),
											   ((float)z-resolution/2)*simulationParams.particleRadius}};// + randRange(-0.01f, 0.01f)}};
				vel[(x * resolution + y) * resolution + z] = (cl_float3) {{0.0f, 0.0f, 0.0f}};
			}
		}
	}
	
	/* Create a device and context */
	device = create_device();
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	if(err < 0) {
		perror("Couldn't create a context");
		exit(1);   
	}

	/* Create data buffers */
	posBuffer = clCreateBuffer (context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
					sizeof(cl_float3) * num_particles, pos, &err);
	if(err < 0) {
		perror("Couldn't create position buffer");
		exit(1);   
	};
	
	velBuffer = clCreateBuffer (context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
				    	sizeof(cl_float3) * num_particles, vel, &err);
	if(err < 0) {
		perror("Couldn't create velocity buffer");
		exit(1);   
	};
	
	densityBuffer = clCreateBuffer (context, CL_MEM_READ_WRITE, 
				   	sizeof(cl_float) * num_particles, NULL, &err);
	if(err < 0) {
		perror("Couldn't create density buffer");
		exit(1);   
	};
	
	densityGuessBuffer = clCreateBuffer (context, CL_MEM_READ_WRITE, 
				   	sizeof(cl_float) * num_particles, NULL, &err);
	if(err < 0) {
		perror("Couldn't create density guess buffer");
		exit(1);   
	};	
	
	densityErrorBuffer = clCreateBuffer (context, CL_MEM_READ_WRITE, 
				   	sizeof(cl_float) * num_particles, NULL, &err);
	if(err < 0) {
		perror("Couldn't create density error buffer");
		exit(1);   
	};
	
	normalBuffer = clCreateBuffer (context, CL_MEM_READ_WRITE, 
				   	sizeof(cl_float3) * num_particles, NULL, &err);
	if(err < 0) {
		perror("Couldn't create normals buffer");
		exit(1);   
	};	
	
	nonPressureForcesBuffer = clCreateBuffer (context, CL_MEM_READ_WRITE, 
				    	sizeof(cl_float3) * num_particles, NULL, &err);
	if(err < 0) {
		perror("Couldn't create non-pressure forces buffer");
		exit(1);   
	};
	
	pressureBuffer = clCreateBuffer (context, CL_MEM_READ_WRITE, 
				    	sizeof(cl_float4) * num_particles, NULL, &err);
	if(err < 0) {
		perror("Couldn't create pressure buffer");
		exit(1);   
	};
	
	coefficientBuffer = clCreateBuffer (context, CL_MEM_READ_WRITE, 
				    	sizeof(cl_float4) * num_particles, NULL, &err);
	if(err < 0) {
		perror("Couldn't create coefficient buffer");
		exit(1);   
	};
	
	dijBuffer = clCreateBuffer (context, CL_MEM_READ_WRITE, 
				    	sizeof(cl_float3) * num_particles, NULL, &err);
	if(err < 0) {
		perror("Couldn't create dij buffer");
		exit(1);   
	};
	
	
	neighbourBuffer = clCreateBuffer (context, CL_MEM_READ_WRITE, 
				    	sizeof(size_t) * num_particles * 400, NULL, &err);
	if(err < 0) {
		perror("Couldn't create neighbour buffer");
		exit(1);   
	};

	/* Build the program and create the kernels */
	program = build_program(context, device, PROGRAM_FILE);

	nonPressureForces = clCreateKernel(program, "nonPressureForces", &err);
	if(err < 0) {
		perror("Couldn't create nonPressureForces kernel");
		exit(1);   
	};

	initPressure = clCreateKernel(program, "initPressure", &err);
	if(err < 0) {
		perror("Couldn't create initPressure kernel");
		exit(1);   
	};

	updateDensity = clCreateKernel(program, "updateDensity", &err);
	if(err < 0) {
		perror("Couldn't create updateDensity kernel");
		exit(1);   
	};

	updateNormals = clCreateKernel(program, "updateNormals", &err);
	if(err < 0) {
		perror("Couldn't create updateNormals kernel");
		exit(1);   
	};
	
	updateDij = clCreateKernel(program, "updateDij", &err);
	if(err < 0) {
		perror("Couldn't create updateDij kernel");
		exit(1);   
	};
	
	updateVelocity = clCreateKernel(program, "timeStep", &err);
	if(err < 0) {
		perror("Couldn't create updateVelocity kernel");
		exit(1);   
	};
	
	updatePositionVelocity = clCreateKernel(program, "timeStep", &err);
	if(err < 0) {
		perror("Couldn't create updatePositionVelocity kernel");
		exit(1);   
	};


	predictDensityPressure = clCreateKernel(program, "predictDensityPressure", &err);
	if(err < 0) {
		perror("Couldn't create predictDensityPressure kernel");
		exit(1);   
	};

	pressureForces = clCreateKernel(program, "pressureForces", &err);
	if(err < 0) {
		perror("Couldn't create pressureForces kernel");
		exit(1);   
	};

	findNeighbours = clCreateKernel(program, "findNeighbours", &err);
	if(err < 0) {
		perror("Couldn't create findNeighbours kernel");
		exit(1);   
	};

	/* Set kernel args */
	err  = clSetKernelArg (nonPressureForces, 0, sizeof(cl_mem), &posBuffer);
	err |= clSetKernelArg (nonPressureForces, 1, sizeof(cl_mem), &velBuffer);
	err |= clSetKernelArg (nonPressureForces, 2, sizeof(cl_mem), &nonPressureForcesBuffer);
	err |= clSetKernelArg (nonPressureForces, 3, sizeof(cl_mem), &densityBuffer);
	err |= clSetKernelArg (nonPressureForces, 4, sizeof(cl_mem), &normalBuffer);
	err |= clSetKernelArg (nonPressureForces, 5, sizeof(cl_mem), &neighbourBuffer);
	err |= clSetKernelArg (nonPressureForces, 6, sizeof(struct fluidParams), &simulationParams);

	if(err < 0) {
		perror("Couldn't set nonPressureForces kernel args");
		exit(1);   
	};


	err  = clSetKernelArg (initPressure, 0, sizeof(cl_mem), &posBuffer);
	err |= clSetKernelArg (initPressure, 1, sizeof(cl_mem), &velBuffer);
	err |= clSetKernelArg (initPressure, 2, sizeof(cl_mem), &densityBuffer);
	err |= clSetKernelArg (initPressure, 3, sizeof(cl_mem), &densityGuessBuffer);
	err |= clSetKernelArg (initPressure, 4, sizeof(cl_mem), &pressureBuffer);
	err |= clSetKernelArg (initPressure, 5, sizeof(cl_mem), &coefficientBuffer);
	err |= clSetKernelArg (initPressure, 6, sizeof(cl_mem), &neighbourBuffer);
	err |= clSetKernelArg (initPressure, 7, sizeof(struct fluidParams), &simulationParams);
	
	if(err < 0) {
		perror("Couldn't set initPressure kernel args");
		exit(1);   
	};


	err  = clSetKernelArg (updateDensity, 0, sizeof(cl_mem), &posBuffer);
	err |= clSetKernelArg (updateDensity, 1, sizeof(cl_mem), &densityBuffer);
	err |= clSetKernelArg (updateDensity, 2, sizeof(cl_mem), &neighbourBuffer);
	err |= clSetKernelArg (updateDensity, 3, sizeof(struct fluidParams), &simulationParams);
	
	if(err < 0) {
		perror("Couldn't set updateDensity kernel args");
		exit(1);   
	};


	err  = clSetKernelArg (updateNormals, 0, sizeof(cl_mem), &posBuffer);
	err |= clSetKernelArg (updateNormals, 1, sizeof(cl_mem), &densityBuffer);
	err |= clSetKernelArg (updateNormals, 2, sizeof(cl_mem), &normalBuffer);
	err |= clSetKernelArg (updateNormals, 3, sizeof(cl_mem), &neighbourBuffer);
	err |= clSetKernelArg (updateNormals, 4, sizeof(struct fluidParams), &simulationParams);
	
	if(err < 0) {
		perror("Couldn't set updateNormals kernel args");
		exit(1);   
	};


	err  = clSetKernelArg (updateDij, 0, sizeof(cl_mem), &posBuffer);
	err |= clSetKernelArg (updateDij, 1, sizeof(cl_mem), &densityBuffer);
	err |= clSetKernelArg (updateDij, 2, sizeof(cl_mem), &pressureBuffer);
	err |= clSetKernelArg (updateDij, 3, sizeof(cl_mem), &dijBuffer);
	err |= clSetKernelArg (updateDij, 4, sizeof(cl_mem), &neighbourBuffer);
	err |= clSetKernelArg (updateDij, 5, sizeof(struct fluidParams), &simulationParams);
	
	if(err < 0) {
		perror("Couldn't set updateNormals kernel args");
		exit(1);   
	};


	err  = clSetKernelArg (updateVelocity, 0, sizeof(cl_mem), NULL);
	err |= clSetKernelArg (updateVelocity, 1, sizeof(cl_mem), &velBuffer);
	err |= clSetKernelArg (updateVelocity, 2, sizeof(cl_mem), NULL);
	err |= clSetKernelArg (updateVelocity, 3, sizeof(cl_mem), &velBuffer);
	err |= clSetKernelArg (updateVelocity, 4, sizeof(cl_mem), &nonPressureForcesBuffer);
	err |= clSetKernelArg (updateVelocity, 5, sizeof(struct fluidParams), &simulationParams);

	if(err < 0) { 
		perror("Couldn't set updateVelocity kernel args");
		exit(1);   
	};


	err  = clSetKernelArg (updatePositionVelocity, 0, sizeof(cl_mem), &posBuffer);
	err |= clSetKernelArg (updatePositionVelocity, 1, sizeof(cl_mem), &velBuffer);
	err |= clSetKernelArg (updatePositionVelocity, 2, sizeof(cl_mem), &posBuffer);
	err |= clSetKernelArg (updatePositionVelocity, 3, sizeof(cl_mem), &velBuffer);
	err |= clSetKernelArg (updatePositionVelocity, 4, sizeof(cl_mem), &nonPressureForcesBuffer);
	err |= clSetKernelArg (updatePositionVelocity, 5, sizeof(struct fluidParams), &simulationParams);

	if(err < 0) { 
		perror("Couldn't set updatePositionVelocity kernel args");
		exit(1);   
	};


	err  = clSetKernelArg (predictDensityPressure, 0, sizeof(cl_mem), &posBuffer);
	err |= clSetKernelArg (predictDensityPressure, 1, sizeof(cl_mem), &densityBuffer);
	err |= clSetKernelArg (predictDensityPressure, 2, sizeof(cl_mem), &densityGuessBuffer);
	err |= clSetKernelArg (predictDensityPressure, 3, sizeof(cl_mem), &densityErrorBuffer);
	err |= clSetKernelArg (predictDensityPressure, 4, sizeof(cl_mem), &pressureBuffer);
	err |= clSetKernelArg (predictDensityPressure, 5, sizeof(cl_mem), &coefficientBuffer);
	err |= clSetKernelArg (predictDensityPressure, 6, sizeof(cl_mem), &dijBuffer);
	err |= clSetKernelArg (predictDensityPressure, 7, sizeof(cl_mem), &neighbourBuffer);
	err |= clSetKernelArg (predictDensityPressure, 8, sizeof(struct fluidParams), &simulationParams);
	
	if(err < 0) {
		perror("Couldn't set predictDensityPressure kernel args");
		exit(1);   
	};


	err  = clSetKernelArg (pressureForces, 0, sizeof(cl_mem), &posBuffer);
	err |= clSetKernelArg (pressureForces, 1, sizeof(cl_mem), &densityBuffer);
	err |= clSetKernelArg (pressureForces, 2, sizeof(cl_mem), &pressureBuffer);
	err |= clSetKernelArg (pressureForces, 3, sizeof(cl_mem), &neighbourBuffer);
	err |= clSetKernelArg (pressureForces, 4, sizeof(struct fluidParams), &simulationParams);
	
	if(err < 0) {
		perror("Couldn't set pressureForces kernel args");
		exit(1);   
	};

	err  = clSetKernelArg (findNeighbours, 0, sizeof(cl_mem), &posBuffer);
	err |= clSetKernelArg (findNeighbours, 1, sizeof(cl_mem), &neighbourBuffer);
	err |= clSetKernelArg (findNeighbours, 2, sizeof(struct fluidParams), &simulationParams);
	
	if(err < 0) {
		perror("Couldn't set updateOldForces kernel args");
		exit(1);   
	};


	/* Create a command queue */
	queue = clCreateCommandQueue(context, device, 0, &err);
	if(err < 0) {
		perror("Couldn't create a command queue");
		exit(1);   
	};   
	
	double iters = 0;
	/* Execute kernels, read data and print */
	for (unsigned int i = 0; i < 500*outputStep; i++)
	{	
		/* Update neighbours */
		err = clEnqueueNDRangeKernel(
				queue,
				findNeighbours,
				1,
				NULL,
				&work_units,
				NULL,
				0,
				NULL,
				NULL);
		if(err < 0) {
			perror("Couldn't enqueue findNeighbours kernel");
			exit(1);   
		};   
		
		/* Update density */
		err = clEnqueueNDRangeKernel(
				queue,
				updateDensity,
				1,
				NULL,
				&work_units,
				NULL,
				0,
				NULL,
				NULL);
		if(err < 0) {
			perror("Couldn't enqueue updateDensity kernel");
			exit(1);   
		};   
		
		/* Update fluid normals */
		err = clEnqueueNDRangeKernel(
				queue,
				updateNormals,
				1,
				NULL,
				&work_units,
				NULL,
				0,
				NULL,
				NULL);
		if(err < 0) {
			perror("Couldn't enqueue updateNormals kernel");
			exit(1);   
		};   	

		/* Calculate non-pressure forces */
		err = clEnqueueNDRangeKernel(
				queue,
				nonPressureForces,
				1,
				NULL,
				&work_units,
				NULL,
				0,
				NULL,
				NULL);
		if(err < 0) {
			printf("Iteration %u\n", i);
			perror("Couldn't enqueue nonPressureForces kernel");
			exit(1);   
		};   
		
		/* Update positions and velocities */
		err = clEnqueueNDRangeKernel(
				queue,
				updateVelocity,
				1,
				NULL,
				&work_units,
				NULL,
				0,
				NULL,
				NULL);
		if(err < 0) {
			printf("Iteration %u\n", i);
			perror("Couldn't enqueue updateVelocity kernel");
			exit(1);   
		};   
		
		/* Update density and pressure, and calculate IISPH coefficients */
		err = clEnqueueNDRangeKernel(
				queue,
				initPressure,
				1,
				NULL,
				&work_units,
				NULL,
				0,
				NULL,
				NULL);
		if(err < 0) {
			printf("Iteration %u\n", i);
			perror("Couldn't enqueue initPressure kernel");
			exit(1);   
		};   
	
		/* Pressure loop */
		for(size_t l=0; l < 100; l++)
		{
			/* Calculate dij */
			err = clEnqueueNDRangeKernel(
					queue,
					updateDij,
					1,
					NULL,
					&work_units,
					NULL,
					0,
					NULL,
					NULL);
			if(err < 0) {
				printf("Iteration %u\n", i);
				perror("Couldn't enqueue updateDij kernel");
				exit(1);   
			};   
	
			err = clEnqueueBarrier(queue);

			/* Predict density and its error and update pressure */
			err = clEnqueueNDRangeKernel(
					queue,
					predictDensityPressure,
					1,
					NULL,
					&work_units,
					NULL,
					0,
					NULL,
					NULL);
			if(err < 0) {
				printf("Iteration %u\n", i);
				perror("Couldn't enqueue predictDensityPressure kernel");
				exit(1);   
			};   

			if (l>2)
			{
				/* Read density error buffer */
				err = clEnqueueReadBuffer(
						queue,
						densityErrorBuffer,
						CL_TRUE,
						0,
						sizeof(cl_float) * num_particles,
						densityError,
						0,
						NULL,
						NULL);
	
				if(err < 0) {
					perror("Couldn't enqueue read");
					exit(1);   
				};

				if (!checkDensityErrors(densityError, num_particles))
				{
					iters += (double)l;
					break;
				}
			}
		}
	
		/* Compute pressure force */
		err = clEnqueueNDRangeKernel(
				queue,
				pressureForces,
				1,
				NULL,
				&work_units,
				NULL,
				0,
				NULL,
				NULL);
		if(err < 0) {
			printf("Iteration %u\n", i);
			perror("Couldn't enqueue pressureForces kernel");
			exit(1);   
		};   
		
		/* Update positions and velocities */
		err = clEnqueueNDRangeKernel(
				queue,
				updatePositionVelocity,
				1,
				NULL,
				&work_units,
				NULL,
				0,
				NULL,
				NULL);
		if(err < 0) {
			printf("Iteration %u\n", i);
			perror("Couldn't enqueue updatePositionVelocity kernel");
			exit(1);   
		};   
		
		if (!(i % outputStep))
		{
			err = clEnqueueReadBuffer(
					queue,
					posBuffer,
					CL_TRUE,
					0,
					sizeof(cl_float3) * num_particles,
					pos,
					0,
					NULL,
					NULL);
	
			if(err < 0) {
				perror("Couldn't enqueue read");
				exit(1);   
			};
	
			printf("%u\t", i);
			for (size_t j = 0; j < num_particles; j++)
			{
				printf("%.4f\t%.4f\t%.4f\t", pos[j].x, pos[j].y, pos[j].z);
			}
			printf("\n");
		}	
	}

	/* Deallocate resources */
	clReleaseKernel(nonPressureForces);
	clReleaseKernel(initPressure);

	clReleaseMemObject(posBuffer);
	clReleaseMemObject(velBuffer);
	clReleaseMemObject(densityBuffer);
	clReleaseMemObject(nonPressureForcesBuffer);
	clReleaseMemObject(pressureBuffer);

	clReleaseCommandQueue(queue);
	clReleaseProgram(program);
	clReleaseContext(context);

	printf("Average iters: %f\n", iters / (500*outputStep));

	return 0;
}

