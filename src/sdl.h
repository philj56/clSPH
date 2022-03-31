/*
 * Copyright (C) 2017-2020 Philip Jones
 *
 * Licensed under the MIT License.
 * See either the LICENSE file, or:
 *
 * https://opensource.org/licenses/MIT
 *
 */

#ifndef CLSPH_SDL_H
#define CLSPH_SDL_H

#include "particle.h"
#include <epoxy/gl.h>
#include <SDL2/SDL.h>
#include <stdbool.h>

struct clsph_sdl {
	SDL_Window *window;
	SDL_GLContext *context;

	GLuint vbo;
	GLuint vao;
	GLuint shader;
};

void sdl_initialise(struct clsph_sdl *sdl, struct particle *particles, size_t num_particles);
bool sdl_update(struct clsph_sdl *sdl, struct particle *particles, size_t num_particles);

#endif /* CLSPH_SDL_H */
