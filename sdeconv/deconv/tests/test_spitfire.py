"""Unit testing the Spitfire deconvolution implementation"""
import os
import numpy as np
from skimage.io import imread

from sdeconv.data import celegans, pollen_poison_noise_blurred, pollen_psf
from sdeconv.deconv import Spitfire
from sdeconv.psfs import SPSFGaussian


def test_spitfire_2d():
    """Unit testing Spitfire 2D deconvolution"""
    root_dir = os.path.dirname(os.path.abspath(__file__))
    image = celegans()

    psf_generator = SPSFGaussian((1.5, 1.5), (15, 15))
    psf = psf_generator()

    filter_ = Spitfire(psf, weight=0.6, reg=0.995, gradient_step=0.01, precision=1e-7, pad=13)
    out_image = filter_(image)

    # imsave(os.path.join(root_dir, 'celegans_spitfire.tif'), out_image.detach().cpu().numpy())
    ref_image = imread(os.path.join(root_dir, 'celegans_spitfire.tif'))

    np.testing.assert_almost_equal(out_image.detach().cpu().numpy()/10, ref_image/10, decimal=0)


def test_spitfire_3d():
    """Unit testing Spitfire 3D deconvolution"""
    root_dir = os.path.dirname(os.path.abspath(__file__))
    image = pollen_poison_noise_blurred()
    psf = pollen_psf()

    filter_ = Spitfire(psf, weight=0.6, reg=0.99995, gradient_step=0.01, precision=1e-7)
    out_image = filter_(image)

    # imsave(os.path.join(root_dir, 'pollen_spitfire.tif'), out_image.detach().cpu().numpy())
    ref_image = imread(os.path.join(root_dir, 'pollen_spitfire.tif'))

    np.testing.assert_almost_equal(out_image.detach().cpu().numpy(), ref_image, decimal=0)
