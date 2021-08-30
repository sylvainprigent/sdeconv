/// \file sgaussianpsf.h
/// \brief sgaussianpsf definitions
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#ifndef sl_gaussian_psf_H
#define sl_gaussian_psf_H

namespace SImg{

void gaussian_psf_2d(float* buffer_out, unsigned int sx, unsigned int sy, float sigma_x, float sigma_y);
void gaussian_psf_3d(float* buffer_out, unsigned int sx, unsigned int sy, unsigned int sz, float sigma_x, float sigma_y, float sigma_z);

}
#endif /* !sl_gaussian_psf_H */
