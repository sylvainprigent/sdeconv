cdef extern from "swiener.h" namespace "SImg":

    void wiener_deconv_2d(float* buffer_in, float* psf, float* buffer_out, unsigned int sx, unsigned int sy,
                          const float& lambda_, const int& connectivity)

    void wiener_deconv_3d(float* buffer_in, float* psf, float* buffer_out, unsigned int sx, unsigned int sy,
                          unsigned int sz, const float& lambda_, const int& connectivity)

