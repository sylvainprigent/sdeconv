/// \file sl_richardson_lucy.c
/// \brief sl_richardson_lucy definitions
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include "srichardsonlucy.h"
#include "SFFT.h"
#include "SShift.h"
#include <fftw3.h>
#include <stdlib.h>
#include "math.h"

#include <iostream>

#ifdef SL_USE_OPENMP
#include "omp.h"
#endif

namespace SImg
{

    void richardsonlucy_2d(float *buffer_in, float *psf, float *buffer_out, unsigned int sx, unsigned int sy, unsigned int niter)
    {

        // memory
        unsigned int n = sx * sy;
        unsigned int n_fft = sx * (sy / 2 + 1);
        float scale = 1.0 / float(n_fft);
        fftwf_complex *fft_in = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * unsigned(n_fft));
        fftwf_complex *fft_out = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * unsigned(n_fft));
        fftwf_complex *fft_psf = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * unsigned(n_fft));
        fftwf_complex *fft_psf_mirror = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * unsigned(n_fft));
        fftwf_complex *fft_tmp = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * unsigned(n_fft));
        float *psf_mirror = (float *)malloc(sizeof(float) * n);
        float *tmp = (float *)malloc(sizeof(float) * n);

        // initialization
        fft2D(buffer_in, fft_in, sx, sy);

#pragma omp parallel for
        for (int p = 0; p < int(n); p++)
        {
            buffer_out[p] = 0.5;
        }
        float *psf_shifted = new float[sx * sy];
        shift2D(psf, psf_shifted, sx, sy, int(-float(sx) / 2.0), int(-float(sy) / 2.0));
        fft2D(psf_shifted, fft_psf, sx, sy);
        delete[] psf_shifted;

// flip psf
#pragma omp parallel for
        for (int x = 0; x < int(sx); x++)
        {
            for (unsigned int y = 0; y < sy; y++)
            {
                psf_mirror[sy * x + y] = psf[sy * x + (sy - 1 - y)];
            }
        }
        fft2D(psf_mirror, fft_psf_mirror, sx, sy);

        unsigned int iter = 0;
        while (iter < niter)
        {
            iter++;
            // tmp = convolve(buffer_out, psf)
            fft2D(buffer_out, fft_out, sx, sy);
#pragma omp parallel for
            for (int p = 0; p < int(n_fft); p++)
            {
                fft_tmp[p][0] = (fft_out[p][0] * fft_psf[p][0] - fft_out[p][1] * fft_psf[p][1]) * scale;
                fft_tmp[p][1] = (fft_out[p][1] * fft_psf[p][0] + fft_out[p][0] * fft_psf[p][1]) * scale;
            }
            ifft2D(fft_tmp, tmp, sx, sy);

// tmp = buffer_in / tmp
#pragma omp parallel for
            for (int p = 0; p < int(n); p++)
            {
                if (tmp[p] > 1e-9)
                {
                    tmp[p] = buffer_in[p] / tmp[p];
                }
                else
                {
                    tmp[p] = 0;
                }
            }
            // im_deconv *= convolve(tmp, psf_mirror)
            fft2D(tmp, fft_tmp, sx, sy);
#pragma omp parallel for
            for (int p = 0; p < int(n_fft); p++)
            {
                fft_tmp[p][0] = (fft_tmp[p][0] * fft_psf[p][0] - fft_tmp[p][1] * fft_psf[p][1]) * scale;
                fft_tmp[p][1] = (fft_tmp[p][1] * fft_psf[p][0] + fft_tmp[p][0] * fft_psf[p][1]) * scale;
            }
            ifft2D(fft_tmp, tmp, sx, sy);

#pragma omp parallel for
            for (int p = 0; p < int(n); p++)
            {
                buffer_out[p] *= tmp[p];
            }
        }

        fftwf_free(fft_in);
        fftwf_free(fft_out);
        fftwf_free(fft_psf);
        fftwf_free(fft_psf_mirror);
        fftwf_free(fft_tmp);
        free(psf_mirror);
        free(tmp);
    }

    void richardsonlucy_tv_2d(float *buffer_in, float *psf, float *buffer_out, unsigned int sx, unsigned int sy, unsigned int niter, float lambda)
    {

        // memory
        unsigned int n = sx * sy;
        unsigned int n_fft = sx * (sy / 2 + 1);
        float scale = 1.0 / float(n_fft);
        fftwf_complex *fft_in = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * unsigned(n_fft));
        fftwf_complex *fft_out = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * unsigned(n_fft));
        fftwf_complex *fft_psf = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * unsigned(n_fft));
        fftwf_complex *fft_psf_mirror = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * unsigned(n_fft));
        fftwf_complex *fft_tmp = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * unsigned(n_fft));
        float *psf_mirror = (float *)malloc(sizeof(float) * n);
        float *tmp = (float *)malloc(sizeof(float) * n);

        float *grad_x = (float *)malloc(sizeof(float) * n);
        float *grad_y = (float *)malloc(sizeof(float) * n);

        // initialization
        fft2D(buffer_in, fft_in, sx, sy);

#pragma omp parallel for
        for (int p = 0; p < int(n); p++)
        {
            buffer_out[p] = 0.5;
        }
        float *psf_shifted = new float[sx * sy];
        shift2D(psf, psf_shifted, sx, sy, int(-float(sx) / 2.0), int(-float(sy) / 2.0));
        fft2D(psf_shifted, fft_psf, sx, sy);
        delete[] psf_shifted;

// flip psf
#pragma omp parallel for
        for (int x = 0; x < int(sx); x++)
        {
            for (unsigned int y = 0; y < sy; y++)
            {
                psf_mirror[sy * x + y] = psf[sy * x + (sy - 1 - y)];
            }
        }
        fft2D(psf_mirror, fft_psf_mirror, sx, sy);

        unsigned int iter = 0;
        while (iter < niter)
        {
            iter++;
            // tmp = convolve(buffer_out, psf)
            fft2D(buffer_out, fft_out, sx, sy);
#pragma omp parallel for
            for (int p = 0; p < int(n_fft); p++)
            {
                fft_tmp[p][0] = (fft_out[p][0] * fft_psf[p][0] - fft_out[p][1] * fft_psf[p][1]) * scale;
                fft_tmp[p][1] = (fft_out[p][1] * fft_psf[p][0] + fft_out[p][0] * fft_psf[p][1]) * scale;
            }
            ifft2D(fft_tmp, tmp, sx, sy);

// tmp = buffer_in / tmp
#pragma omp parallel for
            for (int p = 0; p < int(n); p++)
            {
                if (tmp[p] > 1e-9)
                {
                    tmp[p] = buffer_in[p] / tmp[p];
                }
                else
                {
                    tmp[p] = 0;
                }
            }
            // im_deconv *= convolve(tmp, psf_mirror)
            fft2D(tmp, fft_tmp, sx, sy);
#pragma omp parallel for
            for (int p = 0; p < int(n_fft); p++)
            {
                fft_tmp[p][0] = (fft_tmp[p][0] * fft_psf[p][0] - fft_tmp[p][1] * fft_psf[p][1]) * scale;
                fft_tmp[p][1] = (fft_tmp[p][1] * fft_psf[p][0] + fft_tmp[p][0] * fft_psf[p][1]) * scale;
            }
            ifft2D(fft_tmp, tmp, sx, sy);

// out *= tmp * 1 / ( 1 + lambda* l1(grad(grad/normgrad)))
#pragma omp parallel for
            for (int x = 0; x < int(sx - 1); x++)
            {
                for (unsigned int y = 0; y < sy - 1; y++)
                {
                    float gx = buffer_out[sy * (x) + y] - buffer_out[sy * (x + 1) + y];
                    float gy = buffer_out[sy * x + y] - buffer_out[sy * x + y + 1];
                    float norm = sqrt(gx * gx + gy * gy);
                    if (norm < 1e-9)
                    {
                        grad_x[sy * x + y] = 0.0;
                        grad_y[sy * x + y] = 0.0;
                    }
                    else
                    {
                        grad_x[sy * x + y] = gx / norm;
                        grad_y[sy * x + y] = gy / norm;
                    }
                }
            }
#pragma omp parallel for
            for (int x = 0; x < int(sx - 1); x++)
            {
                for (unsigned int y = 0; y < sy - 1; y++)
                {
                    float gx = grad_x[sy * x + y] - grad_x[sy * (x + 1) + y];
                    float gy = grad_x[sy * x + y] - grad_x[sy * x + y + 1];
                    buffer_out[sy * x + y] *= tmp[sy * x + y] * (1.0 / (1.0 + lambda * (fabs(gx) + fabs(gy))));
                }
            }
        }

        fftwf_free(fft_in);
        fftwf_free(fft_out);
        fftwf_free(fft_psf);
        fftwf_free(fft_psf_mirror);
        fftwf_free(fft_tmp);
        free(psf_mirror);
        free(tmp);
        free(grad_x);
        free(grad_y);
    }

    void richardson_lucy_3d(float *buffer_in, float *psf, float *buffer_out, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int niter)
    {
        // memory
        unsigned int n = sx * sy * sz;
        unsigned int n_fft = sx * sy * (sz / 2 + 1);
        float scale = 1.0 / float(n_fft);
        fftwf_complex *fft_in = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * unsigned(n_fft));
        fftwf_complex *fft_out = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * unsigned(n_fft));
        fftwf_complex *fft_psf = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * unsigned(n_fft));
        fftwf_complex *fft_psf_mirror = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * unsigned(n_fft));
        fftwf_complex *fft_tmp = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * unsigned(n_fft));
        float *psf_mirror = (float *)malloc(sizeof(float) * n);
        float *tmp = (float *)malloc(sizeof(float) * n);

        // initialization
        fft3D(buffer_in, fft_in, sx, sy, sz);

#pragma omp parallel for
        for (int p = 0; p < int(n); p++)
        {
            buffer_out[p] = 0.5;
        }
        float *psf_shifted = new float[sx * sy * sz];
        shift3D(psf, psf_shifted, sx, sy, sz, int(-float(sx) / 2.0), int(-float(sy) / 2.0), int(-float(sz) / 2.0));
        fft3D(psf_shifted, fft_psf, sx, sy, sz);
        delete[] psf_shifted;

// flip psf
        for (int x = 0; x < sx; x++)
        {
            for (int y = 0; y < sy; y++)
            {
                for (int z = 0; z < sz; z++)
                {
                    psf_mirror[z + sz * (y + sy * x)] = psf[(sz - 1 - z) + sz * ((sy - 1 - y) + sy * (sx - 1 - x))];
                    //psf_mirror[z + sz * (y + sy * x)] = psf[(z) + sz * ((sy - 1 - y) + sy * (x))];
                }
            }
        }
        fft3D(psf_mirror, fft_psf_mirror, sx, sy, sz);

        unsigned int iter = 0;
        while (iter < niter)
        {
            iter++;
            // tmp = convolve(buffer_out, psf)
            fft3D(buffer_out, fft_out, sx, sy, sz);
#pragma omp parallel for
            for (int p = 0; p < int(n_fft); p++)
            {
                fft_tmp[p][0] = (fft_out[p][0] * fft_psf[p][0] - fft_out[p][1] * fft_psf[p][1]) * scale;
                fft_tmp[p][1] = (fft_out[p][1] * fft_psf[p][0] + fft_out[p][0] * fft_psf[p][1]) * scale;
            }
            ifft3D(fft_tmp, tmp, sx, sy, sz);

// tmp = buffer_in / tmp
#pragma omp parallel for
            for (int p = 0; p < int(n); p++)
            {
                if (tmp[p] > 1e-9)
                {
                    tmp[p] = buffer_in[p] / tmp[p];
                }
                else
                {
                    tmp[p] = 0;
                }
            }
            // im_deconv *= convolve(tmp, psf_mirror)
            fft3D(tmp, fft_tmp, sx, sy, sz);
#pragma omp parallel for
            for (int p = 0; p < int(n_fft); p++)
            {
                fft_tmp[p][0] = (fft_tmp[p][0] * fft_psf[p][0] - fft_tmp[p][1] * fft_psf[p][1]) * scale;
                fft_tmp[p][1] = (fft_tmp[p][1] * fft_psf[p][0] + fft_tmp[p][0] * fft_psf[p][1]) * scale;
            }
            ifft3D(fft_tmp, tmp, sx, sy, sz);

#pragma omp parallel for
            for (int p = 0; p < int(n); p++)
            {
                buffer_out[p] *= tmp[p];
            }
        }

        fftwf_free(fft_in);
        fftwf_free(fft_out);
        fftwf_free(fft_psf);
        fftwf_free(fft_psf_mirror);
        fftwf_free(fft_tmp);
        free(psf_mirror);
        free(tmp);
    }

}