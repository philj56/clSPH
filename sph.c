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

bool checkDensityErrors(cl_float *densityError)
{
	bool result = false;
	cl_float max = 0;
	for (size_t i = 0; i < 1000; i++)
	{
		if (densityError[i] > max)
			max = densityError[i];
		if (densityError[i] > 10)
			result = true;
	}
	//printf("max: %f\n", max);
	return result;
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
	cl_kernel verletStep;
	cl_kernel verletGuessStep;
	cl_kernel predictDensityPressure;
	cl_kernel pressureForces;
	cl_kernel updateOldForces;

	/* Data */
	size_t num_particles = 1000;
	size_t work_units = num_particles;
	const size_t outputStep = 1;

	struct fluidParams simulationParams;
	
	cl_float3 pos[num_particles];
	cl_float3 vel[num_particles];
	cl_float densityError[num_particles];
	cl_float3 zerosFloat3[1000] = {0};

	/* Buffers */
	cl_mem posBuffer;
	cl_mem posGuessBuffer;
	cl_mem velBuffer;
	cl_mem velGuessBuffer;
	cl_mem densityBuffer;
	cl_mem densityErrorBuffer;
	cl_mem nonPressureForcesBuffer;
	cl_mem pressureBuffer;
	cl_mem oldForcesBuffer;

	srandom(time(NULL));

	/* Setup input data */	
	/* Use SI units */
	{
		const cl_float  particleMass = 0.05f;
		const cl_float  particleRadius = 0.0368f;
		const cl_float  restDensity = 1000.0f;
		const cl_float  interactionRadius = 0.0736842f;
		const cl_float  timeStep = 0.002f;
		const cl_float  viscosity = 0.0001f;
		const cl_float3 gravity = {{0.0f, 0.0f, 0.0f}};
		simulationParams = newFluidParams (particleMass, 
						   particleRadius, 
						   restDensity,
						   interactionRadius,
						   timeStep,
						   viscosity,
						   gravity);
	}

	for (int x = -5; x < 5; x++)
	{
		for (int y = -5; y < 5; y++)
		{
			for (int z = -5; z < 5; z++)
			{
				pos[(x+5) * 100 + (y+5) * 10 + (z+5)] = (cl_float3) {{x*0.1f, y*0.1f, z*0.1f}};
				vel[(x+5) * 100 + (y+5) * 10 + (z+5)] = (cl_float3) {{-x*0.01f, -y*0.01f, -z*0.01f}};
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
	
	posGuessBuffer = clCreateBuffer (context, CL_MEM_READ_WRITE, 
					sizeof(cl_float3) * num_particles, NULL, &err);
	if(err < 0) {
		perror("Couldn't create position guess buffer");
		exit(1);   
	};
	
	velBuffer = clCreateBuffer (context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
				    	sizeof(cl_float3) * num_particles, vel, &err);
	if(err < 0) {
		perror("Couldn't create velocity buffer");
		exit(1);   
	};
	
	
	velGuessBuffer = clCreateBuffer (context, CL_MEM_READ_WRITE, 
				    	sizeof(cl_float3) * num_particles, NULL, &err);
	if(err < 0) {
		perror("Couldn't create velocity guess buffer");
		exit(1);   
	};
	
	densityBuffer = clCreateBuffer (context, CL_MEM_READ_WRITE, 
				   	sizeof(cl_float) * num_particles, NULL, &err);
	if(err < 0) {
		perror("Couldn't create density buffer");
		exit(1);   
	};
	
	densityErrorBuffer = clCreateBuffer (context, CL_MEM_READ_WRITE, 
				   	sizeof(cl_float) * num_particles, NULL, &err);
	if(err < 0) {
		perror("Couldn't create density error buffer");
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
	
	oldForcesBuffer = clCreateBuffer (context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
				    	sizeof(cl_float3) * num_particles, zerosFloat3, &err);
	if(err < 0) {
		perror("Couldn't create old forces buffer");
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
	
	verletStep = clCreateKernel(program, "verletStep", &err);
	if(err < 0) {
		perror("Couldn't create verletStep kernel");
		exit(1);   
	};

	verletGuessStep = clCreateKernel(program, "verletStep", &err);
	if(err < 0) {
		perror("Couldn't create verletGuessStep kernel");
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

	updateOldForces = clCreateKernel(program, "updateOldForces", &err);
	if(err < 0) {
		perror("Couldn't create updateOldForces kernel");
		exit(1);   
	};

	/* Set kernel args */
	err  = clSetKernelArg (nonPressureForces, 0, sizeof(cl_mem), &posBuffer);
	err |= clSetKernelArg (nonPressureForces, 1, sizeof(cl_mem), &velBuffer);
	err |= clSetKernelArg (nonPressureForces, 2, sizeof(cl_mem), &nonPressureForcesBuffer);
	err |= clSetKernelArg (nonPressureForces, 3, sizeof(cl_mem), &densityBuffer);
	err |= clSetKernelArg (nonPressureForces, 4, sizeof(struct fluidParams), &simulationParams);

	if(err < 0) {
		perror("Couldn't set nonPressureForces kernel args");
		exit(1);   
	};


	err = clSetKernelArg (initPressure, 0, sizeof(cl_mem), &pressureBuffer);
	
	if(err < 0) {
		perror("Couldn't set initPressure kernel args");
		exit(1);   
	};


	err  = clSetKernelArg (updateDensity, 0, sizeof(cl_mem), &posBuffer);
	err |= clSetKernelArg (updateDensity, 1, sizeof(cl_mem), &densityBuffer);
	err |= clSetKernelArg (updateDensity, 2, sizeof(struct fluidParams), &simulationParams);
	
	if(err < 0) {
		perror("Couldn't set updateDensity kernel args");
		exit(1);   
	};
	

	err  = clSetKernelArg (verletStep, 0, sizeof(cl_mem), &posBuffer);
	err |= clSetKernelArg (verletStep, 1, sizeof(cl_mem), &velBuffer);
	err |= clSetKernelArg (verletStep, 2, sizeof(cl_mem), &posBuffer);
	err |= clSetKernelArg (verletStep, 3, sizeof(cl_mem), &velBuffer);
	err |= clSetKernelArg (verletStep, 4, sizeof(cl_mem), &nonPressureForcesBuffer);
	err |= clSetKernelArg (verletStep, 5, sizeof(cl_mem), &pressureBuffer);
	err |= clSetKernelArg (verletStep, 6, sizeof(cl_mem), &oldForcesBuffer);
	err |= clSetKernelArg (verletStep, 7, sizeof(struct fluidParams), &simulationParams);

	if(err < 0) {
		perror("Couldn't set verletStep kernel args");
		exit(1);   
	};
	

	err  = clSetKernelArg (verletGuessStep, 0, sizeof(cl_mem), &posBuffer);
	err |= clSetKernelArg (verletGuessStep, 1, sizeof(cl_mem), &velBuffer);
	err |= clSetKernelArg (verletGuessStep, 2, sizeof(cl_mem), &posGuessBuffer);
	err |= clSetKernelArg (verletGuessStep, 3, sizeof(cl_mem), &velGuessBuffer);
	err |= clSetKernelArg (verletGuessStep, 4, sizeof(cl_mem), &nonPressureForcesBuffer);
	err |= clSetKernelArg (verletGuessStep, 5, sizeof(cl_mem), &pressureBuffer);
	err |= clSetKernelArg (verletGuessStep, 6, sizeof(cl_mem), &oldForcesBuffer);
	err |= clSetKernelArg (verletGuessStep, 7, sizeof(struct fluidParams), &simulationParams);

	if(err < 0) {
		perror("Couldn't set verletGuessStep kernel args");
		exit(1);   
	};


	err  = clSetKernelArg (predictDensityPressure, 0, sizeof(cl_mem), &posGuessBuffer);
	err |= clSetKernelArg (predictDensityPressure, 1, sizeof(cl_mem), &densityBuffer);
	err |= clSetKernelArg (predictDensityPressure, 2, sizeof(cl_mem), &densityErrorBuffer);
	err |= clSetKernelArg (predictDensityPressure, 3, sizeof(cl_mem), &pressureBuffer);
	err |= clSetKernelArg (predictDensityPressure, 4, sizeof(struct fluidParams), &simulationParams);
	
	if(err < 0) {
		perror("Couldn't set predictDensityPressure kernel args");
		exit(1);   
	};


	err  = clSetKernelArg (pressureForces, 0, sizeof(cl_mem), &posBuffer);
	err |= clSetKernelArg (pressureForces, 1, sizeof(cl_mem), &densityBuffer);
	err |= clSetKernelArg (pressureForces, 2, sizeof(cl_mem), &pressureBuffer);
	err |= clSetKernelArg (pressureForces, 3, sizeof(struct fluidParams), &simulationParams);
	
	if(err < 0) {
		perror("Couldn't set pressureForces kernel args");
		exit(1);   
	};

	err  = clSetKernelArg (updateOldForces, 0, sizeof(cl_mem), &nonPressureForcesBuffer);
	err |= clSetKernelArg (updateOldForces, 1, sizeof(cl_mem), &pressureBuffer);
	err |= clSetKernelArg (updateOldForces, 2, sizeof(cl_mem), &oldForcesBuffer);
	
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

	/* Initialise Densities */
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
		perror("Couldn't enqueue initial updateDensity kernel");
		exit(1);   
	};   

/*

	// Synchronise 
	err = clEnqueueBarrier(queue);
	if(err < 0) {
	//	printf("Iteration %u\n", i);
		perror("Couldn't enqueue pre-loop barrier");
		exit(1);   
	};   

	clFinish(queue);
	
	err = clEnqueueReadBuffer(
			queue,
			densityBuffer,
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
	for (size_t j = 0; j < num_particles; j++)
	{
		printf("%.3f\t", densityError[j]);
	}
	printf("\n");

	return 0;
*/

	printf("GGG");

	/* Execute kernels, read data and print */
	for (unsigned int i = 0; i < 500*outputStep; i++)
	{	
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
		
		/* Initialise pressure */
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
	
		/* Synchronise */
		err = clEnqueueBarrier(queue);
		if(err < 0) {
			printf("Iteration %u\n", i);
			perror("Couldn't enqueue pre-loop barrier");
			exit(1);   
		};   

		/* Prediction-correction pressure loop */
		for(size_t n = 0; n < 3 || checkDensityErrors(densityError); n++)
		{
			/* Predict positions & velocities */
			err = clEnqueueNDRangeKernel(
					queue,
					verletGuessStep,
					1,
					NULL,
					&work_units,
					NULL,
					0,
					NULL,
					NULL);
			if(err < 0) {
				printf("Iteration %u\n", i);
				perror("Couldn't enqueue verletGuessStep kernel");
				exit(1);   
			};   
		
			/* Synchronise */
			err = clEnqueueBarrier(queue);
			if(err < 0) {
				printf("Iteration %u\n", i);
				perror("Couldn't enqueue first loop barrier");
				exit(1);   
			};   

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
		
			/* Synchronise */
			err = clEnqueueBarrier(queue);
			if(err < 0) {
				printf("Iteration %u\n", i);
				perror("Couldn't enqueue second loop barrier");
				exit(1);   
			};   

			/* Predict density and its error and update pressure */
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
		}
		
		/* Update positions and velocities */
		err = clEnqueueNDRangeKernel(
				queue,
				verletStep,
				1,
				NULL,
				&work_units,
				NULL,
				0,
				NULL,
				NULL);
		if(err < 0) {
			printf("Iteration %u\n", i);
			perror("Couldn't enqueue verletGuessStep kernel");
			exit(1);   
		};   
	
		/* Synchronise */
		err = clEnqueueBarrier(queue);
		if(err < 0) {
			printf("Iteration %u\n", i);
			perror("Couldn't enqueue post-loop barrier");
			exit(1);   
		};   
		
		/* Update old forces */
		err = clEnqueueNDRangeKernel(
				queue,
				updateOldForces,
				1,
				NULL,
				&work_units,
				NULL,
				0,
				NULL,
				NULL);
		if(err < 0) {
			printf("Iteration %u\n", i);
			perror("Couldn't enqueue updateOldForces kernel");
			exit(1);   
		};   
	
		/* Synchronise */
		err = clEnqueueBarrier(queue);
		if(err < 0) {
			printf("Iteration %u\n", i);
			perror("Couldn't enqueue post-loop barrier");
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
				printf("%.3f\t%.3f\t%.3f\t", pos[j].x, pos[j].y, pos[j].z);
			}
			printf("\n");
		}	
	}

	/* Deallocate resources */
	clReleaseKernel(nonPressureForces);
	clReleaseKernel(initPressure);
	clReleaseKernel(verletStep);

	clReleaseMemObject(posBuffer);
	clReleaseMemObject(velBuffer);
	clReleaseMemObject(densityBuffer);
	clReleaseMemObject(nonPressureForcesBuffer);
	clReleaseMemObject(pressureBuffer);
	clReleaseMemObject(oldForcesBuffer);

	clReleaseCommandQueue(queue);
	clReleaseProgram(program);
	clReleaseContext(context);

	return 0;
}

