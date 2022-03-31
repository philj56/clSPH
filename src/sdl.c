#include "log.h"
#include "particle.h"
#include "sdl.h"
#include <epoxy/gl.h>
#include <SDL2/SDL.h>
#include <errno.h>
#include <stdbool.h>

static void gl_initialise(struct clsph_sdl *sdl, struct particle *particles, size_t num_particles);
static void load_shader(GLuint shader, const char *filename);
static GLuint create_shader_program(const char *vert, const char *frag);

void sdl_initialise(struct clsph_sdl *sdl, struct particle *particles, size_t num_particles)
{
	if (SDL_InitSubSystem(SDL_INIT_VIDEO) != 0) {
		log_error("Failed to initialize SDL: %s\n", SDL_GetError());
	}

	/* Main OpenGL settings */
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
	SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);


	sdl->window = SDL_CreateWindow(
			"GBCC",                    // window title
			SDL_WINDOWPOS_UNDEFINED,   // initial x position
			SDL_WINDOWPOS_UNDEFINED,   // initial y position
			640,      // width, in pixels
			480,     // height, in pixels
			SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI// flags
			);

	if (sdl->window == NULL) {
		log_error("Could not create window: %s\n", SDL_GetError());
		SDL_Quit();
		exit(EXIT_FAILURE);
	}

	sdl->context = SDL_GL_CreateContext(sdl->window);
	SDL_GL_MakeCurrent(sdl->window, sdl->context);
	gl_initialise(sdl, particles, num_particles);
}

bool sdl_update(struct clsph_sdl *sdl, struct particle *particles, size_t num_particles)
{
	int width;
	int height;

	SDL_GL_MakeCurrent(sdl->window, sdl->context);
	SDL_GL_GetDrawableSize(sdl->window, &width, &height);

	glViewport(0, 0, width, height);
	const GLfloat view[16] =
	{
		1.0f, 0.0f, 0.0f, 0.0f,
		0.0f, ((float)width / (float)height), 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f
	};

	GLint viewUniform = glGetUniformLocation(sdl->shader, "view");
	glUniformMatrix4fv(viewUniform, 1, GL_TRUE, view);

	glClearColor(0.1, 0.1, 0.1, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glBufferData(GL_ARRAY_BUFFER, num_particles * sizeof(*particles), particles, GL_DYNAMIC_DRAW);
	glDrawArrays(GL_POINTS, 0, num_particles);

	SDL_GL_SwapWindow(sdl->window);

	SDL_Event e;
	while (SDL_PollEvent(&e) != 0) {
		if (e.type == SDL_QUIT) {
			return true;
		} else if (e.type == SDL_WINDOWEVENT) {
			if (e.type == SDL_WINDOWEVENT_CLOSE) {
				return true;
			}
		}
	}

	return false;
}

void gl_initialise(struct clsph_sdl *sdl, struct particle *particles, size_t num_particles)
{
	GLint read_framebuffer = 0;
	GLint draw_framebuffer = 0;
	glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &read_framebuffer);
	glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &draw_framebuffer);

	/* Compile and link the shader programs */
	sdl->shader = create_shader_program(
			SHADER_PATH "vert.vert",
			SHADER_PATH "particle.frag"
			);

	glUseProgram(sdl->shader);

	/* Create a vertex buffer for a quad filling the screen */
	glGenBuffers(1, &sdl->vbo);
	glBindBuffer(GL_ARRAY_BUFFER, sdl->vbo);
	glBufferData(GL_ARRAY_BUFFER, num_particles * sizeof(*particles), particles, GL_DYNAMIC_DRAW);

	/* Create a vertex array and enable vertex attributes for the shaders. */
	glGenVertexArrays(1, &sdl->vao);
	glBindVertexArray(sdl->vao);

	GLint posAttrib = glGetAttribLocation(sdl->shader, "position");
	glVertexAttribPointer(posAttrib, 3, GL_FLOAT, GL_FALSE, sizeof(*particles), 0);
	glEnableVertexAttribArray(posAttrib);

	glBindVertexArray(0);

	const GLfloat model[16] =
	{
		0.5f, 0.0f, 0.0f, 0.0f,
		0.0f, 0.5f, 0.0f, 0.0f,
		0.0f, 0.0f, 0.5f, 0.5f,
		0.0f, 0.0f, 0.0f, 1.0f
	};

	const GLfloat view[16] =
	{
		1.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f
	};

	const GLfloat projection[16] =
	{
		1.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 1.0f
	};

	GLint modelUniform = glGetUniformLocation(sdl->shader, "model");
	glUniformMatrix4fv(modelUniform, 1, GL_TRUE, model);

	GLint viewUniform = glGetUniformLocation(sdl->shader, "view");
	glUniformMatrix4fv(viewUniform, 1, GL_TRUE, view);

	GLint projectionUniform = glGetUniformLocation(sdl->shader, "projection");
	glUniformMatrix4fv(projectionUniform, 1, GL_TRUE, projection);

	/* We don't care about the depth and stencil, so just use a
	 * renderbuffer */
	//glGenRenderbuffers(1, &sdl->rbo);
	//glBindRenderbuffer(GL_RENDERBUFFER, sdl->rbo);
	//glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, GBC_SCREEN_WIDTH, GBC_SCREEN_HEIGHT);
	//glBindRenderbuffer(GL_RENDERBUFFER, 0);

	//glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, sdl->rbo);

	if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
		log_error("Framebuffer is not complete!\n");
		exit(EXIT_FAILURE);
	}
	glBindFramebuffer(GL_READ_FRAMEBUFFER, read_framebuffer);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, draw_framebuffer);

	glEnable(GL_PROGRAM_POINT_SIZE);
	glEnable(GL_DEPTH_TEST);

	/* Bind the actual bits we'll be using to render */
	glBindVertexArray(sdl->vao);
	glUseProgram(sdl->shader);
}

void load_shader(GLuint shader, const char *filename)
{
	errno = 0;
	FILE *fp = fopen(filename, "rb");
	if (!fp) {
		log_error("Failed to load shader %s: %s.\n", filename, strerror(errno));
		exit(EXIT_FAILURE);
	}
	if (fseek(fp, 0, SEEK_END) != 0) {
		log_error("Failed to load shader %s: %s.\n", filename, strerror(errno));
		fclose(fp);
		exit(EXIT_FAILURE);
	}
	long size = ftell(fp);
	if (size <= 0) {
		log_error("Failed to load shader %s: %s.\n", filename, strerror(errno));
		fclose(fp);
		exit(EXIT_FAILURE);
	}
	unsigned long usize = (unsigned long) size;
	GLchar *source = malloc(usize + 1);
	rewind(fp);
	if (fread(source, 1, usize, fp) != usize) {
		log_error("Failed to load shader %s: %s.\n", filename, strerror(errno));
		fclose(fp);
		exit(EXIT_FAILURE);
	}
	fclose(fp);
	source[usize] = '\0';
	glShaderSource(shader, 1, (const GLchar *const *)&source, NULL);
	free(source);

	glCompileShader(shader);

	GLint status;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
	if (status != GL_TRUE) {
		log_error("Failed to compile shader %s!\n", filename);

		GLint info_length = 0;
		glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &info_length);
		if (info_length > 1) {
			char *log = malloc((unsigned)info_length * sizeof(*log));
			glGetShaderInfoLog(shader, info_length, NULL, log);
			log_append_error("%s\n", log);
			free(log);
		}
		exit(EXIT_FAILURE);
	}
}

GLuint create_shader_program(const char *vert, const char *frag)
{
	GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
	load_shader(vertex_shader, vert);

	GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
	load_shader(fragment_shader, frag);

	GLuint shader = glCreateProgram();
	glAttachShader(shader, vertex_shader);
	glAttachShader(shader, fragment_shader);
#ifndef __ANDROID__
	glBindFragDataLocation(shader, 0, "out_colour");
#endif
	glLinkProgram(shader);
	return shader;
}
