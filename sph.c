#define _CRT_SECURE_NO_WARNINGS
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
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
#include "particle.h"
#include "bphys.h"

/* Find a GPU or CPU associated with the first available platform */
cl_device_id create_device() {

	cl_platform_id platform;
	cl_device_id dev;
	int err;

	char name[50];

	/* Identify a platform */
	err = clGetPlatformIDs(1, &platform, NULL);
	if(err < 0) {
		perror("Couldn't identify a platform");
		exit(1);
	} 

	clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(name), &name, NULL);
	printf("Using platform %s\n", name);

	/* Access a device */
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
	if(err == CL_DEVICE_NOT_FOUND) {
		printf("No GPU found, reverting to CPU\n");
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
	err = clBuildProgram(program, 0, NULL, "-I.", NULL, NULL);
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
/*float randRange (float min, float max)
{
	return random() * (max - min) / RAND_MAX + min;
}*/

bool checkDensityErrors(cl_float *densityError, size_t num_particles)
{
//	float max = 0.0f;
//	size_t index = 0;
	size_t num = 0;
	cl_float sum = 0.0f;
	for (size_t i = 0; i < num_particles; i++)
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
	cl_kernel correctKernel;
	cl_kernel updateDensity;
	cl_kernel updateNormals;
	cl_kernel updateDij;
	cl_kernel updatePositionVelocity;
	cl_kernel updateBoundaryVolumes;
	cl_kernel predictDensityPressure;
	cl_kernel copyPressure;
	cl_kernel pressureForces;
	cl_kernel findNeighbours;

	/* Buffers */
	cl_mem particleBuffer;
	cl_mem boundaryBuffer;
	cl_mem densityErrorBuffer;
	cl_mem pressureBuffer;
	cl_mem pressureTempBuffer;
	cl_mem neighbourBuffer;
	cl_mem boundaryNeighbourBuffer;
	
	/* Input data */
	const char*  inputDir = "Blender/Ramp/blendcache_ramp";
	const char*  inputName = "input";
	const size_t inputFrame = 1;

	const char*  boundaryDir = "Blender/Ramp/blendcache_ramp";
	const char*  boundaryName = "boundary";
	const size_t boundaryFrame = 1;

	/* Output Data */
	const char*  cacheDir = "Blender/Ramp/sphcache_ramp";
	const char*  cacheName = "sphramp";
	const size_t num_frames = 400;
	const size_t outputStep = 8;
	const size_t densityCorrectStep = 10008;

	/* Input dependent data */
	size_t num_particles;
	size_t num_boundary_particles;
	struct particle *particles;
	struct boundaryParticle *boundaryParticles;
	cl_float *densityError;

	/* Setup fluid parameters */	
	/* Use SI units where applicable */
	/* gridSize should be the minimum particle interaction radius */
	struct fluidParams simulationParams = 
	{
		.gravity = {{0.0f, 0.0f, -9.81f}},
		.timeStep = 0.002f,
		.relaxationFactor = 0.5f,
		.gridSize = 0.1125f
	};

	{
		/* Some particle types */
		struct particle water =
		{
			.pos                 = {{0.0f}},
			.vel                 = {{0.0f}},
			.velocityAdvection   = {{0.0f}},
			.displacement        = {{0.0f}},
			.sumPressureMovement = {{0.0f}},
			.normal              = {{0.0f}},
			.density             = 0.0f,
			.pressure            = 0.0f,
			.advection           = 0.0f,
			.densityAdvection    = 0.0f,
			.kernelCorrection    = 1.0f,
			.mass                = 0.125f,
			.radius              = 0.05f,
			.restDensity         = 1000.0f,
			.interactionRadius   = 0.1125f,
			.surfaceTension      = 0.0725f,
			.adhesion            = 0.0725f,
			.viscosity           = 0.00089f
		};
		
		struct particle mercury =
		{
			.pos                 = {{0.0f}},
			.vel                 = {{0.0f}},
			.velocityAdvection   = {{0.0f}},
			.displacement        = {{0.0f}},
			.sumPressureMovement = {{0.0f}},
			.normal              = {{0.0f}},
			.density             = 0.0f,
			.pressure            = 0.0f,
			.advection           = 0.0f,
			.densityAdvection    = 0.0f,
			.kernelCorrection    = 1.0f,
			.mass                = 13.56f,
			.radius              = 0.1f,
			.restDensity         = 13560.0f,
			.interactionRadius   = 0.225f,
			.surfaceTension      = 0.4865f,
			.adhesion            = 0.001f,
			.viscosity           = 0.0015f
		};

		struct boundaryParticle boundary = defaultBoundaryParticle;

		boundary.interactionRadius = 0.1125f;

		updateDeducedParams(&water);
		updateDeducedParams(&mercury);
		updateDeducedBoundaryParams(&boundary);
	
		/* Read particle data */
		num_particles = bphysReadFrame(NULL, 0, inputFrame, inputDir, inputName);

		particles = (struct particle*) malloc(sizeof(struct particle) * num_particles);
		densityError = (cl_float*) calloc(num_particles, sizeof(cl_float));

		for (size_t i = 0; i < num_particles; i++)
		{
			particles[i] = water;
		}
		
		bphysReadFrame(particles, num_particles, inputFrame, inputDir, inputName);
		printf("Read %lu particles\n", num_particles);

		/* Read boundary data */
		num_boundary_particles = bphysReadBoundaryFrame(NULL, 0, boundaryFrame, boundaryDir, boundaryName);

		boundaryParticles = (struct boundaryParticle*) malloc(sizeof(struct boundaryParticle) * num_boundary_particles);

		for (size_t i = 0; i < num_boundary_particles; i++)
		{
			boundaryParticles[i] = boundary;
		}
		
		bphysReadBoundaryFrame(boundaryParticles, num_boundary_particles, boundaryFrame, boundaryDir, boundaryName);
		printf("Read %lu boundary particles\n", num_boundary_particles);

	}
		
	
	/* Create a device and context */
	device = create_device();
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	if(err < 0) {
		perror("Couldn't create a context");
		exit(1);   
	}

	/* Create data buffers */
	particleBuffer = clCreateBuffer (context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
					sizeof(struct particle) * num_particles, particles, &err);
	if(err < 0) {
		perror("Couldn't create particle buffer");
		exit(1);   
	};
	
	boundaryBuffer = clCreateBuffer (context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
					sizeof(struct boundaryParticle) * num_boundary_particles, boundaryParticles, &err);
	if(err < 0) {
		perror("Couldn't create boundaryParticle buffer");
		exit(1);   
	};
	
	densityErrorBuffer = clCreateBuffer (context, CL_MEM_WRITE_ONLY, 
				   	sizeof(cl_float) * num_particles, NULL, &err);
	if(err < 0) {
		perror("Couldn't create density error buffer");
		exit(1);   
	};
	
	pressureBuffer = clCreateBuffer (context, CL_MEM_READ_WRITE, 
				    	sizeof(cl_float4) * num_particles, NULL, &err);
	if(err < 0) {
		perror("Couldn't create pressure buffer");
		exit(1);   
	};
	
	pressureTempBuffer = clCreateBuffer (context, CL_MEM_READ_WRITE, 
				    	sizeof(cl_float) * num_particles, NULL, &err);
	if(err < 0) {
		perror("Couldn't create pressure temp buffer");
		exit(1);   
	};
	
	neighbourBuffer = clCreateBuffer (context, CL_MEM_READ_WRITE, 
				    	sizeof(size_t) * num_particles * (MAX_NEIGHBOURS + 1), NULL, &err);
	if(err < 0) {
		perror("Couldn't create neighbour buffer");
		exit(1);   
	};
	
	boundaryNeighbourBuffer = clCreateBuffer (context, CL_MEM_READ_WRITE, 
				    	sizeof(size_t) * num_particles * (MAX_NEIGHBOURS + 1), NULL, &err);
	if(err < 0) {
		perror("Couldn't create boundary neighbour buffer");
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
	
	correctKernel = clCreateKernel(program, "correctKernel", &err);
	if(err < 0) {
		perror("Couldn't create correctKernel kernel");
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
	
	updatePositionVelocity = clCreateKernel(program, "timeStep", &err);
	if(err < 0) {
		perror("Couldn't create updatePositionVelocity kernel");
		exit(1);   
	};
	
	updateBoundaryVolumes = clCreateKernel(program, "updateBoundaryVolumes", &err);
	if(err < 0) {
		perror("Couldn't create updateBoundaryVolumes kernel");
		exit(1);   
	};

	predictDensityPressure = clCreateKernel(program, "predictDensityPressure", &err);
	if(err < 0) {
		perror("Couldn't create predictDensityPressure kernel");
		exit(1);   
	};

	copyPressure = clCreateKernel(program, "copyPressure", &err);
	if(err < 0) {
		perror("Couldn't create copyPressure kernel");
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
	err  = clSetKernelArg (nonPressureForces, 0, sizeof(cl_mem), &particleBuffer);
	err |= clSetKernelArg (nonPressureForces, 1, sizeof(cl_mem), &boundaryBuffer);
	err |= clSetKernelArg (nonPressureForces, 2, sizeof(cl_mem), &neighbourBuffer);
	err |= clSetKernelArg (nonPressureForces, 3, sizeof(cl_mem), &boundaryNeighbourBuffer);
	err |= clSetKernelArg (nonPressureForces, 4, sizeof(struct fluidParams), &simulationParams);

	if(err < 0) {
		perror("Couldn't set nonPressureForces kernel args");
		exit(1);   
	};


	err  = clSetKernelArg (initPressure, 0, sizeof(cl_mem), &particleBuffer);
	err |= clSetKernelArg (initPressure, 1, sizeof(cl_mem), &boundaryBuffer);
	err |= clSetKernelArg (initPressure, 2, sizeof(cl_mem), &neighbourBuffer);
	err |= clSetKernelArg (initPressure, 3, sizeof(cl_mem), &boundaryNeighbourBuffer);
	err |= clSetKernelArg (initPressure, 4, sizeof(struct fluidParams), &simulationParams);
	
	if(err < 0) {
		perror("Couldn't set initPressure kernel args");
		exit(1);   
	};


	err  = clSetKernelArg (correctKernel, 0, sizeof(cl_mem), &particleBuffer);
	err |= clSetKernelArg (correctKernel, 1, sizeof(cl_mem), &boundaryBuffer);
	err |= clSetKernelArg (correctKernel, 2, sizeof(cl_mem), &neighbourBuffer);
	err |= clSetKernelArg (correctKernel, 3, sizeof(cl_mem), &boundaryNeighbourBuffer);
	
	if(err < 0) {
		perror("Couldn't set correctKernel kernel args");
		exit(1);   
	};


	err  = clSetKernelArg (updateDensity, 0, sizeof(cl_mem), &particleBuffer);
	err |= clSetKernelArg (updateDensity, 1, sizeof(cl_mem), &boundaryBuffer);
	err |= clSetKernelArg (updateDensity, 2, sizeof(cl_mem), &neighbourBuffer);
	err |= clSetKernelArg (updateDensity, 3, sizeof(cl_mem), &boundaryNeighbourBuffer);
	
	if(err < 0) {
		perror("Couldn't set updateDensity kernel args");
		exit(1);   
	};


	err  = clSetKernelArg (updateNormals, 0, sizeof(cl_mem), &particleBuffer);
	err |= clSetKernelArg (updateNormals, 1, sizeof(cl_mem), &neighbourBuffer);
	
	if(err < 0) {
		perror("Couldn't set updateNormals kernel args");
		exit(1);   
	};


	err  = clSetKernelArg (updateDij, 0, sizeof(cl_mem), &particleBuffer);
	err |= clSetKernelArg (updateDij, 1, sizeof(cl_mem), &neighbourBuffer);
	err |= clSetKernelArg (updateDij, 2, sizeof(struct fluidParams), &simulationParams);
	
	if(err < 0) {
		perror("Couldn't set updateDij kernel args");
		exit(1);   
	};


	err  = clSetKernelArg (updatePositionVelocity, 0, sizeof(cl_mem), &particleBuffer);
	err |= clSetKernelArg (updatePositionVelocity, 1, sizeof(cl_mem), &pressureBuffer);
	err |= clSetKernelArg (updatePositionVelocity, 2, sizeof(struct fluidParams), &simulationParams);

	if(err < 0) { 
		perror("Couldn't set updatePositionVelocity kernel args");
		exit(1);   
	};


	err  = clSetKernelArg (updateBoundaryVolumes, 0, sizeof(cl_mem), &boundaryBuffer);

	if(err < 0) { 
		perror("Couldn't set updateBoundaryVolumes kernel args");
		exit(1);   
	};


	err  = clSetKernelArg (predictDensityPressure, 0, sizeof(cl_mem), &particleBuffer);
	err |= clSetKernelArg (predictDensityPressure, 1, sizeof(cl_mem), &boundaryBuffer);
	err |= clSetKernelArg (predictDensityPressure, 2, sizeof(cl_mem), &densityErrorBuffer);
	err |= clSetKernelArg (predictDensityPressure, 3, sizeof(cl_mem), &pressureTempBuffer);
	err |= clSetKernelArg (predictDensityPressure, 4, sizeof(cl_mem), &neighbourBuffer);
	err |= clSetKernelArg (predictDensityPressure, 5, sizeof(cl_mem), &boundaryNeighbourBuffer);
	err |= clSetKernelArg (predictDensityPressure, 6, sizeof(struct fluidParams), &simulationParams);
	
	if(err < 0) {
		perror("Couldn't set predictDensityPressure kernel args");
		exit(1);   
	};


	err  = clSetKernelArg (copyPressure, 0, sizeof(cl_mem), &pressureTempBuffer);
	err |= clSetKernelArg (copyPressure, 1, sizeof(cl_mem), &particleBuffer);
	err |= clSetKernelArg (copyPressure, 2, sizeof(cl_mem), &pressureBuffer);
	
	if(err < 0) {
		perror("Couldn't set copyPressure kernel args");
		exit(1);   
	};


	err  = clSetKernelArg (pressureForces, 0, sizeof(cl_mem), &particleBuffer);
	err |= clSetKernelArg (pressureForces, 1, sizeof(cl_mem), &boundaryBuffer);
	err |= clSetKernelArg (pressureForces, 2, sizeof(cl_mem), &neighbourBuffer);
	err |= clSetKernelArg (pressureForces, 3, sizeof(cl_mem), &boundaryNeighbourBuffer);
	err |= clSetKernelArg (pressureForces, 4, sizeof(cl_mem), &pressureBuffer);
	
	if(err < 0) {
		perror("Couldn't set pressureForces kernel args");
		exit(1);   
	};

	err  = clSetKernelArg (findNeighbours, 0, sizeof(cl_mem), &particleBuffer);
	err |= clSetKernelArg (findNeighbours, 1, sizeof(cl_mem), &boundaryBuffer);
	err |= clSetKernelArg (findNeighbours, 2, sizeof(cl_mem), &neighbourBuffer);
	err |= clSetKernelArg (findNeighbours, 3, sizeof(cl_mem), &boundaryNeighbourBuffer);
	err |= clSetKernelArg (findNeighbours, 4, sizeof(size_t), &num_boundary_particles);
	err |= clSetKernelArg (findNeighbours, 5, sizeof(struct fluidParams), &simulationParams);
	
	if(err < 0) {
		perror("Couldn't set findNeighbours kernel args");
		exit(1);   
	};

	/* Create a command queue */
	queue = clCreateCommandQueue(context, device, 0, &err);
	if(err < 0) {
		perror("Couldn't create a command queue");
		exit(1);   
	};   

	/* Initialise neighbours and density */
/*	err = clEnqueueNDRangeKernel(
			queue,
			findNeighbours,
			1,
			NULL,
			&num_particles,
			NULL,
			0,
			NULL,
			NULL);
	if(err < 0) {
		perror("Couldn't enqueue findNeighbours kernel");
		exit(1);   
	};   

	err = clEnqueueNDRangeKernel(
			queue,
			updateDensity,
			1,
			NULL,
			&num_particles,
			NULL,
			0,
			NULL,
			NULL);
	if(err < 0) {
		perror("Couldn't enqueue updateDensity kernel");
		exit(1);   
	};  
*/

	clock_t t = clock();
	/* Calculate boundary particle volumes */
	err = clEnqueueNDRangeKernel(
			queue,
			updateBoundaryVolumes,
			1,
			NULL,
			&num_boundary_particles,
			NULL,
			0,
			NULL,
			NULL);
	if(err < 0) {
		perror("Couldn't enqueue updateBoundaryVolumes kernel");
		exit(1);   
	}; 
	clFinish(queue);
	printf("Finding boundary volumes: %f seconds\n", (float)(clock() - t) / CLOCKS_PER_SEC);
	t = clock();

	double iters = 0;
	/* Execute kernels, read data and print */
	for (unsigned int i = 0; i < num_frames*outputStep; i++)
	{
/*		if (i == 101 * outputStep)
		{
			printf("Gravity change!\n");
			simulationParams.gravity = (cl_float3){{0.0f, 0.0f, -9.81f}};
			err = clSetKernelArg (nonPressureForces, 4, sizeof(struct fluidParams), &simulationParams);
			if(err < 0) {
				perror("Couldn't change gravity");
				exit(1);   
			};
		}
*/		

		/* Update neighbours */
		err = clEnqueueNDRangeKernel(
				queue,
				findNeighbours,
				1,
				NULL,
				&num_particles,
				NULL,
				0,
				NULL,
				NULL);
		if(err < 0) {
			perror("Couldn't enqueue findNeighbours kernel");
			exit(1);   
		};   
		clFinish(queue);
		printf("Finding neighbours: %f seconds\n", (float)(clock() - t) / CLOCKS_PER_SEC);
		t = clock();

		/* Correct kernel and density every densityCorrectStep steps */
		if ((i % densityCorrectStep) == 0 && i > 0)
		{
			err = clEnqueueNDRangeKernel(
					queue,
					correctKernel,
					1,
					NULL,
					&num_particles,
					NULL,
					0,
					NULL,
					NULL);
			if(err < 0) {
				perror("Couldn't enqueue correctKernel kernel");
				exit(1);   
			}  	
		}
		else
		{
			/* Update density */
			err = clEnqueueNDRangeKernel(
					queue,
					updateDensity,
					1,
					NULL,
					&num_particles,
					NULL,
					0,
					NULL,
					NULL);
			if(err < 0) {
				perror("Couldn't enqueue updateDensity kernel");
				exit(1);   
			}  
		}
		clFinish(queue);
		printf("Updating density: %f seconds\n", (float)(clock() - t) / CLOCKS_PER_SEC);
		t = clock();
		/*****************************************/
/*			err = clEnqueueReadBuffer(
					queue,
					particleBuffer,
					CL_TRUE,
					0,
					sizeof(struct particle) * num_particles,
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
				queue,
				updateNormals,
				1,
				NULL,
				&num_particles,
				NULL,
				0,
				NULL,
				NULL);
		if(err < 0) {
			perror("Couldn't enqueue updateNormals kernel");
			exit(1);   
		};   	
		clFinish(queue);
		printf("Updating normals: %f seconds\n", (float)(clock() - t) / CLOCKS_PER_SEC);
		t = clock();

		/* Calculate non-pressure forces */
		err = clEnqueueNDRangeKernel(
				queue,
				nonPressureForces,
				1,
				NULL,
				&num_particles,
				NULL,
				0,
				NULL,
				NULL);
		if(err < 0) {
			printf("Iteration %u\n", i);
			perror("Couldn't enqueue nonPressureForces kernel");
			exit(1);   
		};   
		clFinish(queue);
		printf("Non-pressure forces: %f seconds\n", (float)(clock() - t) / CLOCKS_PER_SEC);
		t = clock();
		
		/* Predict density and pressure, and calculate IISPH coefficients */
		err = clEnqueueNDRangeKernel(
				queue,
				initPressure,
				1,
				NULL,
				&num_particles,
				NULL,
				0,
				NULL,
				NULL);
		if(err < 0) {
			printf("Iteration %u\n", i);
			perror("Couldn't enqueue initPressure kernel");
			exit(1);   
		};   
		clFinish(queue);
		printf("Initialising pressure: %f seconds\n", (float)(clock() - t) / CLOCKS_PER_SEC);
		t = clock();
	
		/* Pressure loop */
		for(size_t l=0; l < 100; l++)
		{
			/* Calculate dij */
			err = clEnqueueNDRangeKernel(
					queue,
					updateDij,
					1,
					NULL,
					&num_particles,
					NULL,
					0,
					NULL,
					NULL);
			if(err < 0) {
				printf("Iteration %u\n", i);
				perror("Couldn't enqueue updateDij kernel");
				exit(1);   
			};   

			/* Predict density and its error and calculate pressure */
			err = clEnqueueNDRangeKernel(
					queue,
					predictDensityPressure,
					1,
					NULL,
					&num_particles,
					NULL,
					0,
					NULL,
					NULL);
			if(err < 0) {
				printf("Iteration %u\n", i);
				perror("Couldn't enqueue predictDensityPressure kernel");
				exit(1);   
			};   

			/* Update pressure */
			err = clEnqueueNDRangeKernel(
					queue,
					copyPressure,
					1,
					NULL,
					&num_particles,
					NULL,
					0,
					NULL,
					NULL);
			if(err < 0) {
				printf("Iteration %u\n", i);
				perror("Couldn't enqueue copyPressure kernel");
				exit(1);   
			};   
			
			if (l>1)
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
		clFinish(queue);
		printf("Pressure loop: %f seconds\n", (float)(clock() - t) / CLOCKS_PER_SEC);
		t = clock();
	
		/* Compute pressure force */
		err = clEnqueueNDRangeKernel(
				queue,
				pressureForces,
				1,
				NULL,
				&num_particles,
				NULL,
				0,
				NULL,
				NULL);
		if(err < 0) {
			printf("Iteration %u\n", i);
			perror("Couldn't enqueue pressureForces kernel");
			exit(1);   
		};   
		clFinish(queue);
		printf("Pressure forces: %f seconds\n", (float)(clock() - t) / CLOCKS_PER_SEC);
		t = clock();
		
		/* Update positions and velocities */
		err = clEnqueueNDRangeKernel(
				queue,
				updatePositionVelocity,
				1,
				NULL,
				&num_particles,
				NULL,
				0,
				NULL,
				NULL);
		if(err < 0) {
			printf("Iteration %u\n", i);
			perror("Couldn't enqueue updatePositionVelocity kernel");
			exit(1);   
		};   
		clFinish(queue);
		printf("Applying pressure: %f seconds\n", (float)(clock() - t) / CLOCKS_PER_SEC);
		t = clock();
		
		/* Write output data */
		if (!(i % outputStep))
		{
			err = clEnqueueReadBuffer(
					queue,
					particleBuffer,
					CL_TRUE,
					0,
					sizeof(struct particle) * num_particles,
					particles,
					0,
					NULL,
					NULL);
	
			if(err < 0) {
				perror("Couldn't enqueue read");
				exit(1);   
			};

			bphysWriteFrame(particles, num_particles, i / outputStep, cacheDir, cacheName);
			printf("Wrote frame %lu\n", i / outputStep);
		}	
		clFinish(queue);
		printf("Writing data: %f seconds\n", (float)(clock() - t) / CLOCKS_PER_SEC);
		t = clock();
	}

	/* Deallocate resources */
	clReleaseKernel(nonPressureForces);
	clReleaseKernel(initPressure);
	clReleaseKernel(updateDensity);
	clReleaseKernel(updateNormals);
	clReleaseKernel(updateDij);
	clReleaseKernel(updatePositionVelocity);
	clReleaseKernel(predictDensityPressure);
	clReleaseKernel(copyPressure);
	clReleaseKernel(pressureForces);
	clReleaseKernel(findNeighbours);

	clReleaseMemObject(particleBuffer);
	clReleaseMemObject(densityErrorBuffer);
	clReleaseMemObject(pressureBuffer);
	clReleaseMemObject(pressureTempBuffer);
	clReleaseMemObject(neighbourBuffer);

	clReleaseCommandQueue(queue);
	clReleaseProgram(program);
	clReleaseContext(context);

	free(particles);
	free(densityError);

	printf("Average iters: %f\n", iters / (num_frames*outputStep));

	return 0;
}

