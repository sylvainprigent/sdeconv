/// \file srichardsonlucy.h
/// \brief srichardsonlucy definitions
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#ifndef srichardsonlucy_H
#define srichardsonlucy_H

namespace SImg{

void richardsonlucy_2d(float* buffer_in, float* psf, float* buffer_out, unsigned int sx, unsigned int sy, unsigned int niter);
void richardsonlucy_tv_2d(float* buffer_in, float* psf, float* buffer_out, unsigned int sx, unsigned int sy, unsigned int niter, float lambda = 0);

void richardson_lucy_3d(float* buffer_in, float* psf, float* buffer_out, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int niter);

}
#endif /* !srichardsonlucy_H */