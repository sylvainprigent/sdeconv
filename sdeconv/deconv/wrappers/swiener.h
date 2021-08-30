/// \file wiener_deconv.h
/// \brief wiener_deconv definitions
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#ifndef sl_wiener_deconv_H
#define sl_wiener_deconv_H

namespace SImg{

void wiener_deconv_2d(float* buffer_in, float* psf, float* buffer_out, unsigned int sx, unsigned int sy, const float& lambda, const int& connectivity = 4);
void wiener_deconv_3d(float* buffer_in, float* psf, float* buffer_out, unsigned int sx, unsigned int sy, unsigned int sz, const float& lambda, const int& connectivity = 4);


void wiener_deconv_airyscan_2d(float** buffer_in, float* psf, float* buffer_out, unsigned int sx, unsigned int sy, unsigned int sd, const float lambda, const int connectivity = 4);

}

#endif /* !sl_wiener_deconv_H */