/// \file wiener_deconv.cpp
/// \brief wiener_deconv implementation
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include "swiener.h"

#include "SFFT.h"
#include "SShift.h"

#include "math.h"
#include <stdlib.h>
#include <iostream>

#include <fftw3.h>

#include "sutils.h"

#ifdef SL_USE_OPENMP
#include "omp.h"
#endif

namespace SImg
{

    void wiener_deconv_2d(float *buffer_in, float *psf, float *buffer_out, unsigned int sx, unsigned int sy, const float& lambda, const int& connectivity)
    {

        // memory initialization
        unsigned int n = sx * sy;
        unsigned int n_fft = sx * (sy / 2 + 1);
        float scale = 1.0 / float(n_fft);
        fftwf_complex *fft_in = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * unsigned(n_fft));
        fftwf_complex *fft_out = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * unsigned(n_fft));
        fftwf_complex *fft_psf = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * unsigned(n_fft));
        fftwf_complex *fft_laplacian = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * unsigned(n_fft));

        // calculate the filter: G
        fft2D(buffer_in, fft_in, sx, sy);
        // H: fft_psf
        float *buffer_psf_shift = (float *)malloc(sizeof(float) * n);
        shift2D(psf, buffer_psf_shift, sx, sy, int(-float(sx) / 2.0), int(-float(sy) / 2.0));
        fft2D(buffer_psf_shift, fft_psf, sx, sy);
        delete buffer_psf_shift;

        // laplacian regularization
        float *buffer_laplacian = (float *)malloc(sizeof(float) * n);
#pragma omp parallel for
        for (int p = 0; p < n; p++)
        {
            buffer_laplacian[p] = 0;
        }
        unsigned int xc = sx / 2;
        unsigned int yc = sy / 2;

        if (connectivity == 4)
        {
            buffer_laplacian[sy * xc + yc] = 4;
            buffer_laplacian[sy * xc + yc - 1] = -1;
            buffer_laplacian[sy * xc + yc + 1] = -1;
            buffer_laplacian[sy * (xc - 1) + yc] = -1;
            buffer_laplacian[sy * (xc + 1) + yc] = -1;
        }
        else if (connectivity == 8)
        {
            buffer_laplacian[sy * xc + yc] = 8;
            buffer_laplacian[sy * (xc - 1) + yc - 1] = -1;
            buffer_laplacian[sy * xc + yc - 1] = -1;
            buffer_laplacian[sy * (xc + 1) + yc - 1] = -1;

            buffer_laplacian[sy * (xc - 1) + yc] = -1;
            buffer_laplacian[sy * (xc + 1) + yc] = -1;

            buffer_laplacian[sy * (xc - 1) + yc + 1] = -1;
            buffer_laplacian[sy * xc + yc + 1] = -1;
            buffer_laplacian[sy * (xc + 1) + yc + 1] = -1;
        }

        fft2D(buffer_laplacian, fft_laplacian, sx, sy);
        delete buffer_laplacian;

#pragma omp parallel for
        for (int p = 0; p < n_fft; p++)
        {
            float den = (pow(fft_psf[p][0], 2) + pow(fft_psf[p][1], 2)) + lambda * (pow(fft_laplacian[p][0], 2) + pow(fft_laplacian[p][1], 2));
            fft_psf[p][0] = fft_psf[p][0] / den;
            fft_psf[p][1] = -fft_psf[p][1] / den;

            fft_out[p][0] = (fft_psf[p][0] * fft_in[p][0] - fft_psf[p][1] * fft_in[p][1]) * scale;
            fft_out[p][1] = (fft_psf[p][1] * fft_in[p][0] + fft_psf[p][0] * fft_in[p][1]) * scale;
        }
        ifft2D(fft_out, buffer_out, sx, sy);
#pragma omp parallel for
        for (int p = 0; p < n; p++)
        {
            if (buffer_out[p] < 0)
            {
                buffer_out[p] = 0;
            }
        }

        fftwf_free(fft_in);
        fftwf_free(fft_out);
        fftwf_free(fft_psf);
        fftwf_free(fft_laplacian);
    }

    void wiener_deconv_3d(float *buffer_in, float *psf, float *buffer_out, unsigned int sx, unsigned int sy, unsigned int sz, const float& lambda, const int& connectivity)
    {
        // memory initialization
        unsigned int n = sx * sy * sz;
        unsigned int n_fft = sx * sy* (sz / 2 + 1);
        float scale = 1.0 / float(n_fft);
        fftwf_complex *fft_in = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * unsigned(n_fft));
        fftwf_complex *fft_out = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * unsigned(n_fft));
        fftwf_complex *fft_psf = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * unsigned(n_fft));
        fftwf_complex *fft_laplacian = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * unsigned(n_fft));

        // calculate the filter: G
        fft3D(buffer_in, fft_in, sx, sy, sz);
        // H: fft_psf
        float *buffer_psf_shift = (float *)malloc(sizeof(float) * n);
        shift3D(psf, buffer_psf_shift, sx, sy, sz, int(-float(sx) / 2.0), int(-float(sy) / 2.0), int(-float(sz) / 2.0));
        fft3D(buffer_psf_shift, fft_psf, sx, sy, sz);
        delete buffer_psf_shift;

        // laplacian regularization
        float *buffer_laplacian = (float *)malloc(sizeof(float) * n);
#pragma omp parallel for
        for (int p = 0; p < n; p++)
        {
            buffer_laplacian[p] = 0;
        }
        unsigned int xc = sx / 2;
        unsigned int yc = sy / 2;
        unsigned int zc = sz / 2;

        if (connectivity == 4)
        {
            buffer_laplacian[zc + sz*(sy * xc + yc)] = 6;
            buffer_laplacian[zc-1 + sz*(sy * xc + yc)] = -1;
            buffer_laplacian[zc+1 + sz*(sy * xc + yc)] = -1;
            buffer_laplacian[zc + sz*(sy * xc + yc - 1)] = -1;
            buffer_laplacian[zc + sz*(sy * xc + yc + 1)] = -1;
            buffer_laplacian[zc + sz*(sy * (xc - 1) + yc)] = -1;
            buffer_laplacian[zc + sz*(sy * (xc + 1) + yc)] = -1;
        }
        else if (connectivity == 8)
        {
            buffer_laplacian[zc + sz*(sy * xc + yc)] = 26;

            buffer_laplacian[zc + sz*(sy * (xc - 1) + yc - 1)] = -1;
            buffer_laplacian[zc + sz*(sy * xc + yc - 1)] = -1;
            buffer_laplacian[zc + sz*(sy * (xc + 1) + yc - 1)] = -1;

            buffer_laplacian[zc + sz*(sy * (xc - 1) + yc)] = -1;
            buffer_laplacian[zc + sz*(sy * (xc + 1) + yc)] = -1;

            buffer_laplacian[zc + sz*(sy * (xc - 1) + yc + 1)] = -1;
            buffer_laplacian[zc + sz*(sy * xc + yc + 1)] = -1;
            buffer_laplacian[zc + sz*(sy * (xc + 1) + yc + 1)] = -1;

            for (int i = -1 ; i <= 1 ; ++i){
                for (int j = -1 ; j <= 1 ; ++j){
                    buffer_laplacian[zc-1 + sz*(sy * (xc + i) + yc + j)] = -1; 
                    buffer_laplacian[zc+1 + sz*(sy * (xc + i) + yc + j)] = -1;            
                }
            }
        }

        fft3D(buffer_laplacian, fft_laplacian, sx, sy, sz);
        delete buffer_laplacian;

#pragma omp parallel for
        for (int p = 0; p < n_fft; p++)
        {
            float den = (pow(fft_psf[p][0], 2) + pow(fft_psf[p][1], 2)) + lambda * (pow(fft_laplacian[p][0], 2) + pow(fft_laplacian[p][1], 2));
            fft_psf[p][0] = fft_psf[p][0] / den;
            fft_psf[p][1] = -fft_psf[p][1] / den;

            fft_out[p][0] = (fft_psf[p][0] * fft_in[p][0] - fft_psf[p][1] * fft_in[p][1]) * scale;
            fft_out[p][1] = (fft_psf[p][1] * fft_in[p][0] + fft_psf[p][0] * fft_in[p][1]) * scale;
        }
        ifft3D(fft_out, buffer_out, sx, sy, sz);
#pragma omp parallel for
        for (int p = 0; p < n; p++)
        {
            if (buffer_out[p] < 0)
            {
                buffer_out[p] = 0;
            }
        }

        fftwf_free(fft_in);
        fftwf_free(fft_out);
        fftwf_free(fft_psf);
        fftwf_free(fft_laplacian);
    }

    void wiener_deconv_airyscan_2d(float **buffer_in, float *psf, float *buffer_out, unsigned int sx, unsigned int sy, unsigned int sd, const float lambda, const int connectivity)
    {
        // memory initialization
        unsigned int n = sx * sy;
        unsigned int n_fft = sx * (sy / 2 + 1);
        float scale = 1.0 / float(n_fft);
        fftwf_complex *fft_in = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * unsigned(n_fft));
        fftwf_complex *fft_out = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * unsigned(n_fft));
        fftwf_complex *fft_psf = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * unsigned(n_fft));
        fftwf_complex *fft_laplacian = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * unsigned(n_fft));

        // H: fft_psf
        float *buffer_psf_shift = (float *)malloc(sizeof(float) * n);
        shift2D(psf, buffer_psf_shift, sx, sy, int(-float(sx) / 2.0), int(-float(sy) / 2.0));
        fft2D(buffer_psf_shift, fft_psf, sx, sy);
        delete buffer_psf_shift;

        // laplacian regularization
        float *buffer_laplacian = (float *)malloc(sizeof(float) * n);
#pragma omp parallel for
        for (int p = 0; p < n; p++)
        {
            buffer_laplacian[p] = 0;
        }
        unsigned int xc = sx / 2;
        unsigned int yc = sy / 2;

        if (connectivity == 4)
        {
            buffer_laplacian[sy * xc + yc] = 4;
            buffer_laplacian[sy * xc + yc - 1] = -1;
            buffer_laplacian[sy * xc + yc + 1] = -1;
            buffer_laplacian[sy * (xc - 1) + yc] = -1;
            buffer_laplacian[sy * (xc + 1) + yc] = -1;
        }
        else if (connectivity == 8)
        {
            buffer_laplacian[sy * xc + yc] = 8;
            buffer_laplacian[sy * (xc - 1) + yc - 1] = -1;
            buffer_laplacian[sy * xc + yc - 1] = -1;
            buffer_laplacian[sy * (xc + 1) + yc - 1] = -1;

            buffer_laplacian[sy * (xc - 1) + yc] = -1;
            buffer_laplacian[sy * (xc + 1) + yc] = -1;

            buffer_laplacian[sy * (xc - 1) + yc + 1] = -1;
            buffer_laplacian[sy * xc + yc + 1] = -1;
            buffer_laplacian[sy * (xc + 1) + yc + 1] = -1;
        }
        fft2D(buffer_laplacian, fft_laplacian, sx, sy);
        delete buffer_laplacian;

        // Calculate denominator
        float *buffer_den = (float *)malloc(sizeof(float) * n_fft);
#pragma omp parallel for
        for (int p = 0; p < n_fft; p++)
        {
            fft_out[p][0] = 0;
            fft_out[p][1] = 0;
            buffer_den[p] = sd * (pow(fft_psf[p][0], 2) + pow(fft_psf[p][1], 2)) + lambda * (pow(fft_laplacian[p][0], 2) + pow(fft_laplacian[p][1], 2));
        }

        // filter in Fourier space
        //#pragma omp parallel for
        for (int d = 0; d < sd; d++)
        {
            fft2D(buffer_in[d], fft_in, sx, sy);
            for (int p = 0; p < n_fft; p++)
            {
                fft_out[p][0] += (fft_psf[p][0] * fft_in[p][0] + fft_psf[p][1] * fft_in[p][1]) * scale / buffer_den[p];
                fft_out[p][1] += (-fft_psf[p][1] * fft_in[p][0] + fft_psf[p][0] * fft_in[p][1]) * scale / buffer_den[p];
            }
        }
        // inverse fourier transform
        ifft2D(fft_out, buffer_out, sx, sy);
#pragma omp parallel for
        for (int p = 0; p < n; p++)
        {
            if (buffer_out[p] < 0)
            {
                buffer_out[p] = 0;
            }
        }

        fftwf_free(fft_in);
        fftwf_free(fft_out);
        fftwf_free(fft_psf);
        fftwf_free(fft_laplacian);
    }

}