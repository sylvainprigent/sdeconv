cdef extern from "spitfire2d.h" namespace "SImg":

    void spitfire2d_deconv_sv(float *blurry_image, unsigned int sx, unsigned int sy, float *psf,
                              float *deconv_image, const float &regularization, const float &weighting,
                              const unsigned int &niter)

    void spitfire2d_deconv_hv(float *blurry_image, unsigned int sx, unsigned int sy, float *psf,
                              float *deconv_image, const float &regularization, const float &weighting,
                              const unsigned int &niter)


cdef extern from "spitfire3d.h" namespace "SImg":

    void spitfire3d_deconv_sv(float *blurry_image, unsigned int sx, unsigned int sy, unsigned int sz,
                              float *psf, float *deconv_image, const float &regularization,
                              const float &weighting, const float &delta, const unsigned int &niter);

    void spitfire3d_deconv_hv(float *blurry_image, unsigned int sx, unsigned int sy, unsigned int sz,
                              float *psf, float *deconv_image, const float &regularization,
                              const float &weighting, const float &delta, const unsigned int &niter);



