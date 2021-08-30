/// \file SFFT.h
/// \brief SFFT functions
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <fftw3.h>

namespace SImg{

fftwf_complex* fft2D(float* inArray, unsigned int sizeX, unsigned int sizeY);
void fft2D(float* in, fftwf_complex* out, unsigned int sizeX, unsigned int sizeY);
float* ifft2D(fftwf_complex* inArray, unsigned int sizeX, unsigned int sizeY);
void ifft2D(fftwf_complex* in, float* out, unsigned int sizeX, unsigned int sizeY);

fftwf_complex* fft3D(float* inArray, unsigned int sizeX, unsigned int sizeY, unsigned int sizeZ);
void fft3D(float* in, fftwf_complex* out, unsigned int sizeX, unsigned int sizeY, unsigned int sizeZ);
float* ifft3D(fftwf_complex* inArray, unsigned int sizeX, unsigned int sizeY, unsigned int sizeZ);
void ifft3D(fftwf_complex* in, float* out, unsigned int sizeX, unsigned int sizeY, unsigned int sizeZ);

float* fftModule(fftwf_complex* inArray, unsigned int sizeX, unsigned int sizeY);
float* fftModule(fftwf_complex* inArray, unsigned int sizeX, unsigned int sizeY, unsigned int sizeZ);

}
