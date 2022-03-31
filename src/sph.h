#ifndef CLSPH_H
#define CLSPH_H

#include "cl.h"
#include "hash.h"
#include "fluid_params.h"
#include "particle.h"
#include "sdl.h"

struct sph {
	size_t num_fluid_particles;
	size_t num_boundary_particles;
	struct particle *particles;
	struct hash *hash_map;

	struct fluid_params params;
	struct clsph_cl cl;
	struct clsph_sdl sdl;
};

#endif /* CLSPH_H */
