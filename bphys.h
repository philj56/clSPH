#ifndef BPHYS_H
#define BPHYS_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "particle.h"

#define MAX_FILENAME 256

void bphysWriteFrame(const struct particle *particles, const unsigned int numParticles, const unsigned int frame, const char* subDir, const char* cacheName)
{
	FILE *file;
	char filename[MAX_FILENAME];

	/* Avoid creating files in root directory if no subdirectory is provided */
	if (strcmp(subDir, "") != 0 && subDir != NULL)
	{
		snprintf(filename, MAX_FILENAME, "%s/%s_%06u_00.bphys", subDir, cacheName, frame);
	}
	else
	{
		snprintf(filename, MAX_FILENAME, "%s_%06u_00.bphys", cacheName, frame);
	}

	file = fopen(filename,"w");
	
	if (file == NULL)
	{
		perror("Couldn't open output file");
		fprintf(stderr, "Hint: make sure subdirectory %s exists\n", subDir);
		exit(-1);
	}

	/* File header data
	 * type = 1 - particle data
	 * data_size = 7 - number of data points per particle (index, pos(x,y,z), vel(x,y,z))
	 */
	static const char bphys[8] = "BPHYSICS";
	static const unsigned int type = 1;
	static const unsigned int data_size = 7;

	/* Write header */
	fwrite(bphys, sizeof(bphys[0]), 8, file);
	fwrite(&type, sizeof(type), 1, file);
	fwrite(&numParticles, sizeof(numParticles), 1, file);
	fwrite(&data_size, sizeof(data_size), 1, file);

	struct particle p;
	for (unsigned int i = 0; i < numParticles; i++)
	{
		p = particles[i];

		/* Write particle data */
		fwrite(&i, sizeof(i), 1, file);
		fwrite(&p.pos.x, sizeof(p.pos.x), 1, file);
		fwrite(&p.pos.y, sizeof(p.pos.y), 1, file);
		fwrite(&p.pos.z, sizeof(p.pos.z), 1, file);
		fwrite(&p.vel.x, sizeof(p.vel.x), 1, file);
		fwrite(&p.vel.y, sizeof(p.vel.y), 1, file);
		fwrite(&p.vel.z, sizeof(p.vel.z), 1, file);
	}

	fclose(file);
}

#endif /* BPHYS_H */
