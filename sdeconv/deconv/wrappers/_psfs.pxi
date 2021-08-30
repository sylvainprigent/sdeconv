cdef extern from "sgibsonlannipsf.h" namespace "SImg":

    void gibson_lanni_psf(float* buffer_out, unsigned int sx, unsigned int sy, unsigned int sz,
                          const float& res_lateral, const float& res_axial, const float& numerical_aperture, 
                          const float& lambd, const float& ti0, const float& ni0, const float& ni,
                          const float& ng0, const float&ng, const float& ns, const float& particle_axial_position)

cdef extern from "sgaussianpsf.h" namespace "SImg":

    void gaussian_psf_2d(float* buffer_out, unsigned int sx, unsigned int sy, float sigma_x, float sigma_y)
    void gaussian_psf_3d(float* buffer_out, unsigned int sx, unsigned int sy, unsigned int sz, float sigma_x,
                         float sigma_y, float sigma_z);
