/*
 * Copyright (C) 2017-2020 Philip Jones
 *
 * Licensed under the MIT License.
 * See either the LICENSE file, or:
 *
 * https://opensource.org/licenses/MIT
 *
 */

#ifndef CLSPH_LOG_H
#define CLSPH_LOG_H

__attribute__((format (printf, 1, 2)))
void log_error(const char *fmt, ...);
__attribute__((format (printf, 1, 2)))
void log_warning(const char *fmt, ...);
__attribute__((format (printf, 1, 2)))
void log_debug(const char *fmt, ...);
__attribute__((format (printf, 1, 2)))
void log_info(const char *fmt, ...);
__attribute__((format (printf, 1, 2)))
void log_append_error(const char *fmt, ...);
__attribute__((format (printf, 1, 2)))
void log_append_warning(const char *fmt, ...);
__attribute__((format (printf, 1, 2)))
void log_append_debug(const char *fmt, ...);
__attribute__((format (printf, 1, 2)))
void log_append_info(const char *fmt, ...);

#endif /* CLSPH_LOG_H */
