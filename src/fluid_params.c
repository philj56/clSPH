#include "fluid_params.h"
#include <math.h>

void update_deduced_params (struct fluid_params *fluid)
{
	fluid->monaghan_spline_normalisation = 1.0f / (M_PI * pow(fluid->interaction_radius * 0.5f, 3.0f));
	fluid->monaghan_spline_prime_normalisation = 10.0f / (7.0f * pow(fluid->interaction_radius * 0.5f, 3.0f));
	fluid->surface_tension_normalisation = 32.0f / (M_PI * pow(fluid->interaction_radius, 9.0f));
	fluid->surface_tension_term = -pow(fluid->interaction_radius, 6.0f) / 64.0f;
	fluid->adhesion_normalisation = 0.007f * pow(fluid->interaction_radius, -3.25);
}
