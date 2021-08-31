/// \file spitfire2d.cpp
/// \brief spitfire2d implementation
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include "spitfire3d.h"

#include "SMath.h"
#include "SShift.h"
#include "SObserverConsole.h"
#include "SException.h"
#include "SNormalize.h"
#include "SFFT.h"
#include <fftw3.h>

#include "math.h"
#ifdef SL_USE_OPENMP
#include "omp.h"
#endif

namespace SImg
{

    void spitfire3d_deconv_sv(float *blurry_image, unsigned int sx, unsigned int sy, unsigned int sz, float *psf, float *deconv_image, const float &regularization, const float &weighting, const float &delta, const unsigned int &niter)
    {
        SObservable* observable = new SObservable();
        SObserverConsole* observer = new SObserverConsole();
        observable->addObserver(observer);
        spitfire3d_deconv_sv(blurry_image, sx, sy, sz, psf, deconv_image, regularization, weighting, delta, niter, true, observable);
        delete observer;
        delete observable;
    }

    void spitfire3d_deconv_hv(float *blurry_image, unsigned int sx, unsigned int sy, unsigned int sz, float *psf, float *deconv_image, const float &regularization, const float &weighting, const float &delta, const unsigned int &niter)
    {
        SObservable* observable = new SObservable();
        SObserverConsole* observer = new SObserverConsole();
        observable->addObserver(observer);
        spitfire3d_deconv_hv(blurry_image, sx, sy, sz, psf, deconv_image, regularization, weighting, delta, niter, true, observable);
        delete observer;
        delete observable;
    }

    void spitfire3d_deconv_sv(float *blurry_image, unsigned int sx, unsigned int sy, unsigned int sz, float *psf, float *deconv_image, const float &regularization, const float &weighting, const float &delta, const unsigned int &niter, bool verbose, SObservable *observable)
    {
#ifdef SL_USE_OPENMP
        omp_set_num_threads(omp_get_max_threads());
        int fftThreads = fftwf_init_threads();
        if (fftThreads == 0)
        {
            std::cout << "Cannot initialize parrallel fft: error " << fftThreads << std::endl;
        }
#endif

        unsigned int N = sx * sy * sz;
        unsigned int Nfft = sx * sy * (sz / 2 + 1);

        // Optical transfer function and its adjoint
        float *OTFReal = new float[N];
        shift3D(psf, OTFReal, sx, sy, sz, int(-float(sx) / 2.0), int(-float(sy) / 2.0), int(-float(sz) / 2.0));

        float *adjoint_PSF = new float[N];
        for (unsigned int x = 0; x < sx; x++)
        {
            for (unsigned int y = 0; y < sy; y++)
            {
                for (unsigned int z = 0; z < sz; z++)
                {
                    adjoint_PSF[z + sz * (y + sy * x)] = psf[(sz - 1 - z) + sz * ((sy - 1 - y) + sy * (sx - 1 - x))];
                }
            }
        }

        float *adjoint_PSF_shift = new float[N];
        shift3D(adjoint_PSF, adjoint_PSF_shift, sx, sy, sz, -(int(sx) - 1) % 2, -(int(sy) - 1) % 2, -int((sz - 1)) % 2);

        float *adjoint_OTFReal = new float[N];
        shift3D(adjoint_PSF_shift, adjoint_OTFReal, sx, sy, sz, int(-float(sx) / 2.0), int(-float(sy) / 2.0), int(-float(sz) / 2.0));
        delete[] adjoint_PSF_shift;

        // Splitting parameters
        float dual_step = SMath::max(0.01, SMath::min(0.1, regularization));
        float primal_step = 0.99 / (0.5 + (12 * pow(weighting, 2.) + pow(1 - weighting, 2.)) * dual_step);
        float primal_weight = primal_step * weighting;
        float primal_weight_comp = primal_step * (1 - weighting);
        float dual_weight = dual_step * weighting;
        float dual_weight_comp = dual_step * (1 - weighting);

        float *dual_images0 = (float *)malloc(sizeof(float) * N);
        float *dual_images1 = (float *)malloc(sizeof(float) * N);
        float *dual_images2 = (float *)malloc(sizeof(float) * N);
        float *dual_images3 = (float *)malloc(sizeof(float) * N);
        float *auxiliary_image = (float *)malloc(sizeof(float) * N);
        float *residue_image = (float *)malloc(sizeof(float) * N);

#pragma omp parallel for
        for (int i = 0; i < int(N); i++)
        {
            dual_images0[i] = 0.0;
            dual_images1[i] = 0.0;
            dual_images2[i] = 0.0;
            dual_images3[i] = 0.0;
            deconv_image[i] = blurry_image[i];
        }

        fftwf_complex *blurry_image_FT = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * unsigned(Nfft));
        fftwf_complex *deconv_image_FT = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * unsigned(Nfft));
        fftwf_complex *residue_image_FT = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * unsigned(Nfft));
        fftwf_complex *OTF = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * unsigned(Nfft));
        fftwf_complex *adjoint_OTF = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * unsigned(Nfft));

        fft3D(blurry_image, blurry_image_FT, sx, sy, sz);
        fft3D(OTFReal, OTF, sx, sy, sz);
        fft3D(adjoint_OTFReal, adjoint_OTF, sx, sy, sz);

        free(OTFReal);
        free(adjoint_OTFReal);

#pragma omp parallel for
        for (int i = 0; i < int(Nfft); i++)
        {
            deconv_image_FT[i][0] = blurry_image_FT[i][0];
            deconv_image_FT[i][1] = blurry_image_FT[i][1];
        }

        float inv_reg = 1.0 / regularization;

#ifdef SL_USE_OPENMP
        observable->notify("Use " + std::to_string(omp_get_max_threads()) + " threads");
        fftwf_plan_with_nthreads(omp_get_max_threads());
#endif

        fftwf_plan Planfft = fftwf_plan_dft_r2c_3d(sx, sy, sz, deconv_image, deconv_image_FT, FFTW_ESTIMATE);
        fftwf_plan Planifft = fftwf_plan_dft_c2r_3d(sx, sy, sz, residue_image_FT, residue_image, FFTW_ESTIMATE);

        // Deconvolution process
        for (unsigned int iter = 0; iter < niter; iter++)
        {
            // Primal optimization
#pragma omp parallel for
            for (int i = 0; i < int(N); i++)
            {
                auxiliary_image[i] = deconv_image[i];
            }
            fftwf_execute(Planfft);

            // Data term
#pragma omp parallel for
            for (int i = 0; i < int(Nfft); i++)
            {
                float real_tmp = OTF[i][0] * deconv_image_FT[i][0] - OTF[i][1] * deconv_image_FT[i][1] - blurry_image_FT[i][0];
                float imag_tmp = OTF[i][0] * deconv_image_FT[i][1] + OTF[i][1] * deconv_image_FT[i][0] - blurry_image_FT[i][1];

                residue_image_FT[i][0] = adjoint_OTF[i][0] * real_tmp - adjoint_OTF[i][1] * imag_tmp;
                residue_image_FT[i][1] = adjoint_OTF[i][0] * imag_tmp + adjoint_OTF[i][1] * real_tmp;
            }

            fftwf_execute(Planifft);

            // gradient term
#pragma omp parallel for
            for (int x = 1; x < int(sx - 1); x++)
            {
                for (unsigned int y = 1; y < sy - 1; y++)
                {
                    for (unsigned int z = 1; z < sz - 1; z++)
                    {
                        unsigned int p = z + sz * (y + sy * x);
                        unsigned int pxm = p - sz * sy;
                        unsigned int pym = p - sz;
                        unsigned int pzm = p - 1;

                        float tmp = deconv_image[p] - primal_step * residue_image[p] / float(N);

                        float dx_adj = dual_images0[pxm] - dual_images0[p];
                        float dy_adj = dual_images1[pym] - dual_images1[p];
                        float dz_adj = delta * (dual_images2[pzm] - dual_images2[p]);

                        tmp -= (primal_weight * (dx_adj + dy_adj + dz_adj) + primal_weight_comp * dual_images3[p]);

                        if (tmp > 1.0)
                        {
                            deconv_image[p] = 1.0;
                        }
                        else if (tmp < 0.0)
                        {
                            deconv_image[p] = 0.0;
                        }
                        else
                        {
                            deconv_image[p] = tmp;
                        }
                    }
                }
            }

            // notify iterations
            if (verbose)
            {
                int iter_n = niter / 10;
                if (iter_n < 1)
                    iter_n = 1;
                if (iter % iter_n == 0)
                {
                    observable->notifyProgress(100 * (float(iter) / float(niter)));
                }
            }

            // Dual optimization
#pragma omp parallel for
            for (int i = 0; i < int(N); i++)
            {
                auxiliary_image[i] = 2 * deconv_image[i] - auxiliary_image[i];
            }

#pragma omp parallel for
            for (int x = 1; x < int(sx - 1); x++)
            {
                for (unsigned int y = 1; y < sy - 1; y++)
                {
                    for (unsigned int z = 1; z < sz - 1; z++)
                    {
                        unsigned int p = z + sz * (y + sy * x);
                        unsigned int pxp = p + sz * sy;
                        unsigned int pyp = p + sz;
                        unsigned int pzp = p + 1;

                        dual_images0[p] += dual_weight * (auxiliary_image[pxp] - auxiliary_image[p]);
                        dual_images1[p] += dual_weight * (auxiliary_image[pyp] - auxiliary_image[p]);
                        dual_images2[p] += dual_weight * (delta * (auxiliary_image[pzp] - auxiliary_image[p]));
                        dual_images3[p] += dual_weight_comp * auxiliary_image[p];
                    }
                }
            }

#pragma omp parallel for
            for (int i = 0; i < int(N); i++)
            {
                float tmp = inv_reg * sqrt(dual_images0[i] * dual_images0[i] + dual_images1[i] * dual_images1[i] + dual_images2[i] * dual_images2[i] + dual_images3[i] * dual_images3[i]);
                if (tmp > 1.0)
                {
                    float inv_tmp = 1.0 / tmp;
                    dual_images0[i] *= inv_tmp;
                    dual_images1[i] *= inv_tmp;
                    dual_images2[i] *= inv_tmp;
                    dual_images3[i] *= inv_tmp;
                }
            }

        } // endfor (int iter = 0; iter < nb_iters_max; iter++)

        // copy output
        free(dual_images0);
        free(dual_images1);
        free(dual_images2);
        free(dual_images3);
        free(auxiliary_image);
        free(residue_image);

        fftwf_free(blurry_image_FT);
        fftwf_free(deconv_image_FT);
        fftwf_free(residue_image_FT);
        fftwf_free(OTF);
        fftwf_free(adjoint_OTF);

        if (verbose)
        {
            observable->notifyProgress(100);
        }
    }

    void spitfire3d_deconv_hv(float *blurry_image, unsigned int sx, unsigned int sy, unsigned int sz, float *psf, float *deconv_image, const float &regularization, const float &weighting, const float &delta, const unsigned int &niter, bool verbose, SObservable *observable)
    {
        int N = sx * sy * sz;
        int Nfft = sx * sy * (sz / 2 + 1);
        float sqrt2 = float(sqrt(2.));

#ifdef SL_USE_OPENMP
        observable->notify("Use " + std::to_string(omp_get_max_threads()) + " threads");
        omp_set_num_threads(omp_get_max_threads());

        int fftThreads = fftwf_init_threads();
        if (fftThreads == 0)
        {
            std::cout << "Cannot initialize parrallel fft: error " << fftThreads << std::endl;
        }
#endif

       // Optical transfer function and its adjoint
        float *OTFReal = new float[N];
        shift3D(psf, OTFReal, sx, sy, sz, int(-float(sx) / 2.0), int(-float(sy) / 2.0), int(-float(sz) / 2.0));

        float *adjoint_PSF = new float[N];
        for (unsigned int x = 0; x < sx; x++)
        {
            for (unsigned int y = 0; y < sy; y++)
            {
                for (unsigned int z = 0; z < sz; z++)
                {
                    adjoint_PSF[z + sz * (y + sy * x)] = psf[(sz - 1 - z) + sz * ((sy - 1 - y) + sy * (sx - 1 - x))];
                }
            }
        }

        float *adjoint_PSF_shift = new float[N];
        shift3D(adjoint_PSF, adjoint_PSF_shift, sx, sy, sz, -(int(sx) - 1) % 2, -(int(sy) - 1) % 2, -int((sz - 1)) % 2);

        float *adjoint_OTFReal = new float[N];
        shift3D(adjoint_PSF_shift, adjoint_OTFReal, sx, sy, sz, int(-float(sx) / 2.0), int(-float(sy) / 2.0), int(-float(sz) / 2.0));
        delete[] adjoint_PSF_shift;

        // Splitting parameters
        float dual_step = SMath::max(0.001, SMath::min(0.01, regularization));
        float primal_step = 0.99 / (0.5 + (144 * pow(weighting, 2.) + pow(1 - weighting, 2.)) * dual_step);
        float primal_weight = primal_step * weighting;
        float primal_weight_comp = primal_step * (1 - weighting);
        float dual_weight = dual_step * weighting;
        float dual_weight_comp = dual_step * (1 - weighting);

        // intermedaite data
        float *dual_images0 = (float *)malloc(sizeof(float) * N);
        float *dual_images1 = (float *)malloc(sizeof(float) * N);
        float *dual_images2 = (float *)malloc(sizeof(float) * N);
        float *dual_images3 = (float *)malloc(sizeof(float) * N);
        float *dual_images4 = (float *)malloc(sizeof(float) * N);
        float *dual_images5 = (float *)malloc(sizeof(float) * N);
        float *dual_images6 = (float *)malloc(sizeof(float) * N);
        float *auxiliary_image = (float *)malloc(sizeof(float) * N);
        float *residue_image = (float *)malloc(sizeof(float) * N);

#pragma omp parallel for
        for (int i = 0; i < int(N); i++)
        {
            dual_images0[i] = 0.0;
            dual_images1[i] = 0.0;
            dual_images2[i] = 0.0;
            dual_images3[i] = 0.0;
            dual_images4[i] = 0.0;
            dual_images5[i] = 0.0;
            dual_images6[i] = 0.0;
            deconv_image[i] = blurry_image[i];
        }

        fftwf_complex *blurry_image_FT = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * unsigned(Nfft));
        fftwf_complex *deconv_image_FT = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * unsigned(Nfft));
        fftwf_complex *residue_image_FT = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * unsigned(Nfft));
        fftwf_complex *OTF = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * unsigned(Nfft));
        fftwf_complex *adjoint_OTF = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * unsigned(Nfft));

        fft3D(blurry_image, blurry_image_FT, sx, sy, sz);
        fft3D(OTFReal, OTF, sx, sy, sz);
        fft3D(adjoint_OTFReal, adjoint_OTF, sx, sy, sz);

        //free(blurry_array);
        free(OTFReal);
        free(adjoint_OTFReal);

#pragma omp parallel for
        for (int i = 0; i < int(Nfft); i++)
        {
            deconv_image_FT[i][0] = blurry_image_FT[i][0];
            deconv_image_FT[i][1] = blurry_image_FT[i][1];
            residue_image_FT[i][0] = blurry_image_FT[i][0];
            residue_image_FT[i][1] = blurry_image_FT[i][1];
        }
        float inv_reg = 1.0 / regularization;

#ifdef SL_USE_OPENMP
        fftwf_plan_with_nthreads(omp_get_max_threads());
#endif

        fftwf_plan Planfft = fftwf_plan_dft_r2c_3d(sx, sy, sz, deconv_image, deconv_image_FT, FFTW_ESTIMATE);
        fftwf_plan Planifft = fftwf_plan_dft_c2r_3d(sx, sy, sz, residue_image_FT, residue_image, FFTW_ESTIMATE);

        // Deconvolution process
        for (unsigned int iter = 0; iter < niter; iter++)
        {
            // Primal optimization
#pragma omp parallel for
            for (int i = 0; i < N; i++)
            {
                auxiliary_image[i] = deconv_image[i];
            }
            fftwf_execute(Planfft);

            // Data term
#pragma omp parallel for
            for (int i = 0; i < Nfft; i++)
            {
                float real_tmp = OTF[i][0] * deconv_image_FT[i][0] - OTF[i][1] * deconv_image_FT[i][1] - blurry_image_FT[i][0];
                float imag_tmp = OTF[i][0] * deconv_image_FT[i][1] + OTF[i][1] * deconv_image_FT[i][0] - blurry_image_FT[i][1];

                residue_image_FT[i][0] = adjoint_OTF[i][0] * real_tmp - adjoint_OTF[i][1] * imag_tmp;
                residue_image_FT[i][1] = adjoint_OTF[i][0] * imag_tmp + adjoint_OTF[i][1] * real_tmp;
            }
            fftwf_execute(Planifft);

            // gradient term
#pragma omp parallel for
            for (int x = 1; x < int(sx - 1); x++)
            {
                for (unsigned int y = 1; y < sy - 1; y++)
                {
                    for (unsigned int z = 1; z < sz - 1; z++)
                    {

                        unsigned int p = z + sz * (y + sy * x);
                        unsigned int pxm = p - sz * sy;
                        unsigned int pym = p - sz;
                        unsigned int pzm = p - 1;
                        unsigned int pxp = p + sz * sy;
                        unsigned int pyp = p + sz;
                        unsigned int pzp = p + 1;

                        float tmp = deconv_image[p] - primal_step * residue_image[p] / float(N);

                        float dxx_adj = dual_images0[pxm] - 2 * dual_images0[p] + dual_images0[pxp];
                        float dyy_adj = dual_images1[pym] - 2 * dual_images1[p] + dual_images1[pyp];
                        float dzz_adj = (delta * delta) * (dual_images2[pzm] - 2 * dual_images2[p] + dual_images2[pzp]);

                        // Other terms
                        float dxy_adj = dual_images3[p] - dual_images3[pxm] - dual_images3[pym] + dual_images3[z + sz * (y - 1 + sy * (x - 1))];
                        float dyz_adj = delta * (dual_images4[p] - dual_images4[pym] - dual_images4[pzm] + dual_images4[z - 1 + sz * (y - 1 + sy * x)]);
                        float dzx_adj = delta * (dual_images5[p] - dual_images5[pzm] - dual_images5[pxm] + dual_images5[z - 1 + sz * (y + sy * (x - 1))]);

                        tmp -= (primal_weight * (dxx_adj + dyy_adj + dzz_adj + sqrt2 * (dxy_adj + dyz_adj + dzx_adj)) + primal_weight_comp * dual_images6[p]);

                        if (tmp > 1.0)
                        {
                            deconv_image[p] = 1.0;
                        }
                        else if (tmp < 0.0)
                        {
                            deconv_image[p] = 0.0;
                        }
                        else
                        {
                            deconv_image[p] = tmp;
                        }
                    }
                }
            }


            // Stopping criterion
            if (verbose)
            {
                int iter_n = niter / 10;
                if (iter_n < 1)
                    iter_n = 1;
                if (iter % iter_n == 0)
                {
                    observable->notifyProgress(100 * (float(iter) / float(niter)));
                }
            }

            // Dual optimization
#pragma omp parallel for
            for (int i = 0; i < N; i++)
            {
                auxiliary_image[i] = 2 * deconv_image[i] - auxiliary_image[i];
            }

#pragma omp parallel for
            for (int x = 1; x < int(sx-1); x++)
            {
                for (unsigned int y = 1; y < sy-1; y++)
                {
                    for (unsigned int z = 1; z < sz-1; z++)
                    {

                        unsigned int p = z + sz * (y + sy * x);  
                        unsigned int pxm = p - sz*sy;
                        unsigned int pym = p - sz;  
                        unsigned int pzm = p - 1;
                        unsigned int pxp = p + sz*sy;
                        unsigned int pyp = p + sz;
                        unsigned int pzp = p + 1;  

                        dual_images0[p] += dual_weight * (auxiliary_image[pxp] - 2 * auxiliary_image[p] + auxiliary_image[pxm]);
                        dual_images1[p] += dual_weight * (auxiliary_image[pyp] - 2 * auxiliary_image[p] + auxiliary_image[pym]);
                        dual_images2[p] += dual_weight * ((delta * delta) * (auxiliary_image[pzp] - 2 * auxiliary_image[p] + auxiliary_image[pzm]));
                        dual_images3[p] += sqrt2 * dual_weight * (auxiliary_image[z + sz * (y + 1 + sy * (x + 1))] - auxiliary_image[pxp] - auxiliary_image[pyp] + auxiliary_image[p]);
                        dual_images4[p] += sqrt2 * dual_weight * (delta * (auxiliary_image[z + 1 + sz * (y + 1 + sy * x)] - auxiliary_image[pyp] - auxiliary_image[pzp] + auxiliary_image[p]));
                        dual_images5[p] += sqrt2 * dual_weight * (delta * (auxiliary_image[z + 1 + sz * (y + sy * (x + 1))] - auxiliary_image[pxp] - auxiliary_image[pzp] + auxiliary_image[p]));
                        dual_images6[p] += dual_weight_comp * auxiliary_image[p];
                    }
                }
            }

#pragma omp parallel for
            for (int i = 0; i < N; i++)
            {
                float tmp = inv_reg * sqrt(dual_images0[i] * dual_images0[i] + dual_images1[i] * dual_images1[i] + dual_images2[i] * dual_images2[i] + 
                                           dual_images3[i] * dual_images3[i] + dual_images4[i] * dual_images4[i] + dual_images4[i] * dual_images4[i] + 
                                           dual_images6[i] * dual_images6[i]);
                if (tmp > 1.0)
                {
                    float inv_tmp = 1.0 / tmp;
                    dual_images0[i] *= inv_tmp;
                    dual_images1[i] *= inv_tmp;
                    dual_images2[i] *= inv_tmp;
                    dual_images3[i] *= inv_tmp;
                    dual_images4[i] *= inv_tmp;
                    dual_images5[i] *= inv_tmp;
                    dual_images6[i] *= inv_tmp;
                }
            }

        } // endfor (int iter = 0; iter < nb_iters_max; iter++)

        // copy output
        free(dual_images0);
        free(dual_images1);
        free(dual_images2);
        free(dual_images3);
        free(dual_images4);
        free(dual_images5);
        free(dual_images6);
        free(auxiliary_image);
        free(residue_image);

        fftwf_free(blurry_image_FT);
        fftwf_free(deconv_image_FT);
        fftwf_free(residue_image_FT);

        if (verbose)
        {
            observable->notifyProgress(100);
        }
    }

    void spitfire3d_deconv(float *blurry_image, unsigned int sx, unsigned int sy, unsigned int sz, float *psf, float *deconv_image, const float &regularization, const float &weighting, const float &delta, const unsigned int &niter, const std::string &method, bool verbose, SObservable *observable)
    {
        // normalize the input image
        unsigned int bs = sx * sy * sz;
        float imin = blurry_image[0];
        float imax = blurry_image[0];
        for (unsigned int i = 1; i < bs; ++i)
        {
            float val = blurry_image[i];
            if (val > imax)
            {
                imax = val;
            }
            if (val < imin)
            {
                imin = val;
            }
        }

        float *blurry_image_norm = new float[bs];
        normL2(blurry_image, sx, sy, sz, 1, 1, blurry_image_norm);

        // run denoising
        if (method == "SV")
        {
            spitfire3d_deconv_sv(blurry_image_norm, sx, sy, sz, psf, deconv_image, regularization, weighting, delta, niter, verbose, observable);
        }
        else if (method == "HV")
        {
            spitfire3d_deconv_hv(blurry_image_norm, sx, sy, sz, psf, deconv_image, regularization, weighting, delta, niter, verbose, observable);
        }
        else
        {
            throw SException("spitfire3d: method must be SV or HV");
        }

        // normalize back intensities
        float omin = deconv_image[0];
        float omax = deconv_image[0];
        for (unsigned int i = 1; i < bs; ++i)
        {
            float val = deconv_image[i];
            if (val > omax)
            {
                omax = val;
            }
            if (val < omin)
            {
                omin = val;
            }
        }

#pragma omp parallel for
        for (int i = 0; i < int(bs); ++i)
        {
            deconv_image[i] = (deconv_image[i] - omin)/(omax-omin);
            deconv_image[i] = deconv_image[i] * (imax - imin) + imin;
        }

        delete[] blurry_image_norm;
    }
}