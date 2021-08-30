import pytest

import os
import numpy as np
from sdeconv import data
from sdeconv.deconv import SpitfireDeconv, PSFGaussian
from skimage.io import imread, imsave


def test_spitfire_hv_2d():

    image = data.celegans()

    psf_gauss = PSFGaussian((1.5, 1.5), image.shape)
    psf_gauss.run()

    regularization = pow(2, -12)
    weighting = 0.6
    model = 'HV'
    niter = 200
    deconv = SpitfireDeconv(regularization, weighting, model, niter)
    deconv_image = deconv.run(image, psf_gauss.psf_)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    ref_file = os.path.join(dir_path, 'deconv_spitfire_hv_celegans.tif')
    # imsave(ref_file, deconv_image)
    ref_image = imread(ref_file)

    assert np.array_equal(deconv_image, ref_image)


def test_spitfire_sv_2d():

    image = data.celegans()

    psf_gauss = PSFGaussian((1.5, 1.5), image.shape)
    psf_gauss.run()

    regularization = pow(2, -12)
    weighting = 0.6
    model = 'SV'
    niter = 200
    deconv = SpitfireDeconv(regularization, weighting, model, niter)
    deconv_image = deconv.run(image, psf_gauss.psf_)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    ref_file = os.path.join(dir_path, 'deconv_spitfire_sv_celegans.tif')
    # imsave(ref_file, deconv_image)
    ref_image = imread(ref_file)

    assert np.array_equal(deconv_image, ref_image)


def test_spitfire_hv_3d():

    image = data.pollen_poison_noise_blurred()
    psf = data.pollen_psf()

    regularization = pow(2, -30)
    weighting = 0.6
    model = 'HV'
    niter = 200
    deconv = SpitfireDeconv(regularization, weighting, model, niter)
    deconv_image = deconv.run(image, psf)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    ref_file = os.path.join(dir_path, 'deconv_spitfire_hv_pollen.tif')
    # imsave(ref_file, deconv_image)
    ref_image = imread(ref_file)

    assert np.array_equal(deconv_image, ref_image)


def test_spitfire_sv_3d():

    image = data.pollen_poison_noise_blurred()
    psf = data.pollen_psf()

    regularization = pow(2, -30)
    weighting = 0.6
    model = 'SV'
    niter = 200
    deconv = SpitfireDeconv(regularization, weighting, model, niter)
    deconv_image = deconv.run(image, psf)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    ref_file = os.path.join(dir_path, 'deconv_spitfire_sv_pollen.tif')
    # imsave(ref_file, deconv_image)
    ref_image = imread(ref_file)

    assert np.array_equal(deconv_image, ref_image)
