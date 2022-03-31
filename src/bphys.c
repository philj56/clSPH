#include "bphys.h"
#include "log.h"
#include "particle.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define MAX_FILENAME 256

void bphys_write_frame(const struct particle *particles, const unsigned int num_particles, const unsigned int frame, const char* sub_dir, const char* cache_name)
{
	FILE *file;
	char filename[MAX_FILENAME];

	/* Avoid creating files in root directory if no subdirectory is provided */
	if (strcmp(sub_dir, "") != 0 && sub_dir != NULL)
	{
		snprintf(filename, MAX_FILENAME, "%s/%s_%06u_00.bphys", sub_dir, cache_name, frame);
	}
	else
	{
		snprintf(filename, MAX_FILENAME, "%s_%06u_00.bphys", cache_name, frame);
	}

	file = fopen(filename,"w");
	
	if (file == NULL)
	{
		log_error("Couldn't open output file");
		log_error("Hint: make sure subdirectory %s exists\n", sub_dir);
		exit(-1);
	}

	/* File header data
	 * data_type = 1 - particle data
	 * data_size = 7 - number of data points per particle (index, pos(x,y,z), vel(x,y,z))
	 */
	static const char bphys[9] = "BPHYSICS";
	static const unsigned int data_type = 1;
	static const unsigned int data_size = 7;

	/* Write header */
	fwrite(bphys, sizeof(bphys[0]), 8, file);
	fwrite(&data_type, sizeof(data_type), 1, file);
	fwrite(&num_particles, sizeof(num_particles), 1, file);
	fwrite(&data_size, sizeof(data_size), 1, file);

	struct particle p;
	for (unsigned int i = 0; i < num_particles; i++)
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

unsigned int bphys_read_frame(struct particle *particles, const unsigned int num_particles, const char *filename)
{
	FILE *file;

	file = fopen(filename, "r");
	
	if (file == NULL)
	{
		log_error("Couldn't open input file");
		exit(-1);
	}

	/* File header data
	 * type = 1 - particle data
	 * data_size = 7 - number of data points per particle (index, pos(x,y,z), vel(x,y,z))
	 */
	static char bphys[8];
	static unsigned int data_type;
	static unsigned int data_size;
	unsigned int num_particles_ret = 0;
	size_t err;

	/* Read header */
	err  = fread(bphys, sizeof(bphys[0]), 8, file);
	if (err != 8)
	{
		if (ferror(file))
		{
			log_error("Error reading %s\n", filename);
		}
		fclose(file);
		return 0;
	}

	err  = fread(&data_type, sizeof(data_type), 1, file);
	err |= fread(&num_particles_ret, sizeof(num_particles_ret), 1, file);
	err |= fread(&data_size, sizeof(data_size), 1, file);
	if (err != 1)
	{
		if (ferror(file))
		{
			log_error("Error reading %s\n", filename);
		}
		fclose(file);
		return 0;
	}

	/* Return number of particles in file if particles is null */
	if (particles == NULL)
	{
		return num_particles_ret;
	}

	/* Read particle data */
	struct particle p;
	if (num_particles > 0)
	{
		for (unsigned int i = 0; i < num_particles; i++)
		{
			p = particles[i];
			unsigned int dummy;
	
			/* Read particle data */
			err  = fread(&dummy, sizeof(dummy), 1, file);
			err |= fread(&p.pos.x, sizeof(p.pos.x), 1, file);
			err |= fread(&p.pos.y, sizeof(p.pos.y), 1, file);
			err |= fread(&p.pos.z, sizeof(p.pos.z), 1, file);
			err |= fread(&p.vel.x, sizeof(p.vel.x), 1, file);
			err |= fread(&p.vel.y, sizeof(p.vel.y), 1, file);
			err |= fread(&p.vel.z, sizeof(p.vel.z), 1, file);

			if (err != 1)
			{
				if (ferror(file))
				{
					log_error("Error reading %s\n", filename);
				}
				fclose(file);
				return i;
			}

			particles[i] = p;
		}
	}
	fclose(file);

	return num_particles;
}

void ply_write_frame(const struct particle *particles, const unsigned int num_particles, const unsigned int frame, const char* sub_dir, const char* cache_name)
{
	FILE *file;
	char filename[MAX_FILENAME];

	/* Avoid creating files in root directory if no subdirectory is provided */
	if (strcmp(sub_dir, "") != 0 && sub_dir != NULL)
	{
		snprintf(filename, MAX_FILENAME, "%s/%s_%06u_00.ply", sub_dir, cache_name, frame);
	}
	else
	{
		snprintf(filename, MAX_FILENAME, "%s_%06u_00.ply", cache_name, frame);
	}

	file = fopen(filename,"w");
	
	if (file == NULL)
	{
		log_error("Couldn't open output file");
		log_error("Hint: make sure subdirectory %s exists\n", sub_dir);
		exit(-1);
	}

	/* File header data
	 * data_type = 1 - particle data
	 * data_size = 7 - number of data points per particle (index, pos(x,y,z), vel(x,y,z))
	 */
	static const char header[] =
		"ply\n"
		"format ascii 1.0\n"
		"element vertex %u\n"
		"property float x\n"
		"property float y\n"
		"property float z\n"
		"end_header\n";

	/* Write header */
	fprintf(file, header, num_particles);

	struct particle p;
	for (unsigned int i = 0; i < num_particles; i++)
	{
		p = particles[i];

		/* Write particle data */
		fprintf(file, "%f %f %f\n", p.pos.x, p.pos.y, p.pos.z);
	}

	fclose(file);
}
