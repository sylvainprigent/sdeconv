/// \file sgaussianpsf.c
/// \brief sgaussianpsf implementation
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020


#include "sgaussianpsf.h"
#include "math.h"

namespace SImg{

void gaussian_psf_2d(float* buffer_out, unsigned int sx, unsigned int sy, float sigma_x, float sigma_y){
    int x0=sx/2;
    int y0=sy/2;
    unsigned int x, y;
    float sigma_x2 = 0.5 / (sigma_x*sigma_x);
    float sigma_y2 = 0.5 / (sigma_y*sigma_y);
    for (x = 0 ; x < sx ; x++){
        for (y = 0 ; y < sy ; y++){
            buffer_out[y + sy*x] = exp( - pow(x-x0,2)*sigma_x2 - pow(y-y0,2)*sigma_y2 );
        }
    }
}

void gaussian_psf_3d(float* buffer_out, unsigned int sx, unsigned int sy, unsigned int sz, float sigma_x, float sigma_y, float sigma_z){
    int x0=sx/2;
    int y0=sy/2;
    int z0=sz/2;
    unsigned int x, y, z;
    float sigma_x2 = 0.5 / sigma_x*sigma_x;
    float sigma_y2 = 0.5 / sigma_y*sigma_y;
    float sigma_z2 = 0.5 / sigma_z*sigma_z;
    for (x = 0 ; x < sx ; x++){
        for (y = 0 ; y < sy ; y++){
            for (z = 0 ; z < sz ; z++){
                buffer_out[z + sz*(y + sy*x)] = exp( - pow(x-x0,2)*sigma_x2 - pow(y-y0,2)*sigma_y2 - pow(z-z0,2)*sigma_z2 );
            }
        }
    }
}

}
