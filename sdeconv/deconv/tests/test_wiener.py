import pytest

import os
import numpy as np
from sdeconv import data
from sdeconv.deconv import WienerDeconv, PSFGaussian
from skimage.io import imread, imsave


def test_wiener_2d():

    image = data.celegans()

    psf_gauss = PSFGaussian((1.5, 1.5), image.shape)
    psf_gauss.run()

    wiener = WienerDeconv(lambda_=0.05)
    deconv_image = wiener.run(image, psf_gauss.psf_)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    ref_file = os.path.join(dir_path, 'deconv_wiener_celegans.tif')
    # imsave(ref_file, deconv_image)
    ref_image = imread(ref_file)

    assert np.array_equal(deconv_image, ref_image)


def test_wiener_3d():

    image = data.pollen_poison_noise_blurred()
    psf = data.pollen_psf()

    wiener = WienerDeconv(lambda_=0.005)
    deconv_image = wiener.run(image, psf)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    ref_file = os.path.join(dir_path, 'deconv_wiener_pollen.tif')
    # imsave(ref_file, deconv_image)
    ref_image = imread(ref_file)

    assert np.array_equal(deconv_image, ref_image)