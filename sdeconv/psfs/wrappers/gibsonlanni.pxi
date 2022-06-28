cdef extern from "_gibsonlanni.h":

    void gibson_lanni_psf(float* buffer_out, unsigned int sx, unsigned int sy, unsigned int sz,
                          const float& res_lateral, const float& res_axial, const float& numerical_aperture,
                          const float& lambd, const float& ti0, const float& ni0, const float& ni,
                          const float& ng0, const float&ng, const float& ns, const float& particle_axial_position)