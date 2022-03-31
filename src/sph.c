#include "bphys.h"
#include "cl.h"
#include "sph.h"
#include "log.h"
#include "particle.h"
#include "sdl.h"

#include <epoxy/gl.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>

static bool force_quit;

static void quit(int sig);

void quit(int sig) 
{
	(void) sig;
	if (force_quit) {
		exit(EXIT_FAILURE);
	}
	force_quit = true;
}

int main(int argc, char *argv[])
{
#ifdef _WIN32
	signal(SIGINT, quit);
#else
	struct sigaction act = {
		.sa_handler = quit
	};
	sigfillset(&act.sa_mask);
	sigaction(SIGINT, &act, NULL);
#endif
	const char *file =  "blender/blendcache_1M/1M_000001_00.bphys";
	struct sph sph = {0};

	sph.num_fluid_particles = bphys_read_frame(NULL, 0, file);
	log_debug("%zu\n", sph.num_fluid_particles);
	sph.particles = calloc(sph.num_fluid_particles, sizeof(*sph.particles));
	bphys_read_frame(sph.particles, sph.num_fluid_particles, file);

	sph.hash_map = calloc(2 * sph.num_fluid_particles, sizeof(*sph.hash_map));
	for (size_t i = 0; i < sph.num_fluid_particles; i++) {
		OPENCL_UINT idx = hash(sph.particles[i].pos, sph.params.interaction_radius, 2 * sph.num_fluid_particles);
		if (sph.hash_map[idx].num_indices == MAX_HASH) {
			continue;
		};
		sph.hash_map[idx].indices[sph.hash_map[idx].num_indices] = i;
		sph.hash_map[idx].num_indices++;
	}

	sph.params = (struct fluid_params) {
		.gravity             = {{0.0f, -0.00f, 0.0f}},
		.time_step            = 0.02f,
		.relaxation_factor    = 0.5f,
		.grid_size            = 0.1125f,
		.mass                = 0.001f * 8*0.125f,
		.radius              = 0.1f * 2*0.05f,
		.rest_density         = 1000.0f,
		.interaction_radius   = 0.1f * 2*0.1125f,
		.surface_tension      = 0.0725f,
		.adhesion            = 0.0725f,
		.viscosity           = 0.00089f
	};
	update_deduced_params(&sph.params);
	cl_initialise(&sph);

	for (size_t i = 0; i < 10; i++) {
		log_debug("{ .x = %f, .y = %f, .z = %f }\n",
				sph.particles[i].pos.x,
				sph.particles[i].pos.y,
				sph.particles[i].pos.z);
	}

	sdl_initialise(&sph.sdl, sph.particles, sph.num_fluid_particles);

	for (size_t i = 0; !force_quit; i++){
		cl_update(&sph);
		if (sdl_update(&sph.sdl, sph.particles, sph.num_fluid_particles)) {
			break;
		}
		ply_write_frame(sph.particles, sph.num_fluid_particles, i, "blender/blendcache_1M", "1M");
	}

	free(sph.particles);
	free(sph.hash_map);
}
