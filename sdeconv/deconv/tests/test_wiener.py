"""Unit testing the Wiener deconvolution implementation"""
import os
import numpy as np
from skimage.io import imread

from sdeconv.data import celegans, pollen_poison_noise_blurred, pollen_psf
from sdeconv.deconv import SWiener
from sdeconv.psfs import SPSFGaussian


def test_wiener_2d():
    """Unit testing wiener 2D deconvolution"""
    root_dir = os.path.dirname(os.path.abspath(__file__))
    image = celegans()

    psf_generator = SPSFGaussian((1.5, 1.5), (13, 13))
    psf = psf_generator()

    filter_ = SWiener(psf, beta=0.005, pad=13)
    out_image = filter_(image)

    # imsave(os.path.join(root_dir, 'celegans_wiener.tif'), out_image.detach().numpy())
    ref_image = imread(os.path.join(root_dir, 'celegans_wiener.tif'))

    np.testing.assert_almost_equal(out_image.detach().cpu().numpy(), ref_image, decimal=1)


def test_wiener_3d():
    """Unit testing wiener 3D deconvolution"""
    root_dir = os.path.dirname(os.path.abspath(__file__))
    image = pollen_poison_noise_blurred()
    psf = pollen_psf()

    filter_ = SWiener(psf, beta=0.0005, pad=(16, 64, 64))
    out_image = filter_(image)

    # imsave(os.path.join(root_dir, 'pollen_wiener.tif'), out_image.detach().numpy())
    ref_image = imread(os.path.join(root_dir, 'pollen_wiener.tif'))

    np.testing.assert_almost_equal(out_image.detach().cpu().numpy(), ref_image, decimal=1)
