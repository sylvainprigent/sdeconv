cdef extern from "srichardsonlucy.h" namespace "SImg":

    void richardsonlucy_2d(float* buffer_in, float* psf, float* buffer_out, unsigned int sx,
                           unsigned int sy, unsigned int niter)

    void richardson_lucy_3d(float* buffer_in, float* psf, float* buffer_out, unsigned int sx,
                            unsigned int sy, unsigned int sz, unsigned int niter)
