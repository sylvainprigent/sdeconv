/// \file spitfire2d.h
/// \brief spitfire2d definitions
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include "SObservable.h"

namespace SImg
{

    /// \brief Sparse total variation deconvolution
    /// \param[in] blurry_image Buffer of the input blurry image
    /// \param[in] sx Number of rows in the blurry image
    /// \param[in] sy Number of columns in the blurry image
    /// \param[in] psf Buffer of the PSF image. Must be same size as blurry_image
    /// \param[out] deconv_image Buffer of the output deblurred image
    /// \param[in] regularization regularization parameter
    /// \param[in] weighting Sparsity weighting in [0,1]. 0 sparse, 1 no sparse
    /// \param[in] niter Number of iteration
    /// \param[in] verbose True to emit progress information
    /// \param[in] observable Pointer to the observable object for verbose
    void spitfire2d_deconv_sv(float *blurry_image, unsigned int sx, unsigned int sy, float *psf, float *deconv_image, const float &regularization, const float &weighting, const unsigned int &niter, bool verbose, SObservable *observable);

    /// \brief use default observable as command line observer
    void spitfire2d_deconv_sv(float *blurry_image, unsigned int sx, unsigned int sy, float *psf, float *deconv_image, const float &regularization, const float &weighting, const unsigned int &niter);


    /// \brief Sparse hessian variation deconvolution
    /// \param[in] blurry_image Buffer of the input blurry image
    /// \param[in] sx Number of rows in the blurry image
    /// \param[in] sy Number of columns in the blurry image
    /// \param[in] psf Buffer of the PSF image. Must be same size as blurry_image
    /// \param[out] deconv_image Buffer of the output deblurred image
    /// \param[in] regularization regularization parameter
    /// \param[in] weighting Sparsity weighting in [0,1]. 0 sparse, 1 no sparse
    /// \param[in] niter Number of iteration
    /// \param[in] verbose True to emit progress information
    /// \param[in] observable Pointer to the observable object for verbose
    void spitfire2d_deconv_hv(float *blurry_image, unsigned int sx, unsigned int sy, float *psf, float *deconv_image, const float &regularization, const float &weighting, const unsigned int &niter, bool verbose, SObservable *observable);

    /// \brief Sparse of hessian variation. This fonction also normalize the input data with it L2 norm
    /// \param[in] blurry_image Buffer of the input blurry image
    /// \param[in] sx Number of rows in the blurry image
    /// \param[in] sy Number of columns in the blurry image
    /// \param[in] psf Buffer of the PSF image. Must be same size as blurry_image
    /// \param[out] deconv_image Buffer of the output deblurred image
    /// \param[in] regularization regularization parameter
    /// \param[in] weighting Sparsity weighting in [0,1]. 0 sparse, 1 no sparse
    /// \param[in] niter Number of iteration
    /// \param[in] method Name of the method "SV" or "HV"
    /// \param[in] verbose True to emit progress information
    /// \param[in] observable Pointer to the observable object for verbose
    void spitfire2d_deconv(float *blurry_image, unsigned int sx, unsigned int sy, float *psf, float *deconv_image, const float &regularization, const float &weighting, const unsigned int &niter, const std::string &method, bool verbose, SObservable *observable);

    /// \brief use default observable as command line observer
    void spitfire2d_deconv_hv(float *blurry_image, unsigned int sx, unsigned int sy, float *psf, float *deconv_image, const float &regularization, const float &weighting, const unsigned int &niter);

}