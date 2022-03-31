/*
 * Copyright (C) 2017-2020 Philip Jones
 *
 * Licensed under the MIT License.
 * See either the LICENSE file, or:
 *
 * https://opensource.org/licenses/MIT
 */

#version 150 core

out vec3 out_colour;

const vec3 light_dir = normalize(vec3(0.5, -0.5, 1.0));
const vec3 light_diffuse = vec3(1.0, 1.0, 1.0);

void main()
{
	vec2 circ_coord = 2 * gl_PointCoord - 1.0;
	if (dot(circ_coord, circ_coord) > 1.0) {
		discard;
	}

	vec3 normal = vec3(circ_coord, sqrt(1 - dot(circ_coord, circ_coord)));
	float c = max(dot(normal, light_dir), 0);
	out_colour = light_diffuse * c;
}
