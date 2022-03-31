#ifndef CLSPH_BPHYS_H
#define CLSPH_BPHYS_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "particle.h"

#define MAX_FILENAME 256

void bphys_write_frame(
		const struct particle *particles,
		const unsigned int num_particles,
		const unsigned int frame,
		const char* sub_dir,
		const char* cache_name);

unsigned int bphys_read_frame(
		struct particle *particles,
		const unsigned int num_particles,
		const char* filename);

void ply_write_frame(
		const struct particle *particles,
		const unsigned int num_particles,
		const unsigned int frame,
		const char* sub_dir,
		const char* cache_name);


#endif /* CLSPH_BPHYS_H */
