/*
 * Copyright (C) 2017-2020 Philip Jones
 *
 * Licensed under the MIT License.
 * See either the LICENSE file, or:
 *
 * https://opensource.org/licenses/MIT
 *
 */

#include <stdarg.h>
#include <stdio.h>

#if defined(_WIN32) || defined(__ANDROID__)
#define RED   ""
#define YEL   ""
#define BLU   ""
#define RESET ""
#else
/* TODO: Clean this up */
#define RED   "\x1B[31m"
//#define GRN   "\x1B[32m"
#define YEL   "\x1B[33m"
#define BLU   "\x1B[34m"
//#define MAG   "\x1B[35m"
//#define CYN   "\x1B[36m"
//#define WHT   "\x1B[37m"
#define RESET "\x1B[0m"
#endif


void log_error(const char *const fmt, ...)
{
	va_list args;
	va_start(args, fmt);
	fprintf(stderr, "[" RED "ERROR" RESET "]: ");
	vfprintf(stderr, fmt, args);
	va_end(args);
}

void log_warning(const char *const fmt, ...)
{
	va_list args;
	va_start(args, fmt);
	fprintf(stderr, "[" YEL "WARNING" RESET "]: ");
	vfprintf(stderr, fmt, args);
	va_end(args);
}

void log_debug(const char *const fmt, ...)
{
#ifndef DEBUG
	return;
#endif
	va_list args;
	va_start(args, fmt);
	printf("[" BLU "DEBUG" RESET "]: ");
	vprintf(fmt, args);
	va_end(args);
}

void log_info(const char *const fmt, ...)
{
	va_list args;
	va_start(args, fmt);
	printf("[INFO]: ");
	vprintf(fmt, args);
	va_end(args);
}

void log_append_error(const char *const fmt, ...)
{
	va_list args;
	va_start(args, fmt);
	vfprintf(stderr, fmt, args);
	va_end(args);
}

void log_append_warning(const char *const fmt, ...)
{
	va_list args;
	va_start(args, fmt);
	vfprintf(stderr, fmt, args);
	va_end(args);
}

void log_append_debug(const char *const fmt, ...)
{
#ifndef DEBUG
	return;
#endif
	va_list args;
	va_start(args, fmt);
	vprintf(fmt, args);
	va_end(args);
}

void log_append_info(const char *const fmt, ...)
{
	va_list args;
	va_start(args, fmt);
	vprintf(fmt, args);
	va_end(args);
}
