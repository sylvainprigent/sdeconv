import pytest

import os
import numpy as np
from sdeconv import data
from sdeconv.deconv import RichardsonLucy, PSFGaussian
from skimage.io import imread, imsave


def test_richardson_lucy_2d():

    image = data.celegans()

    psf_gauss = PSFGaussian((1.5, 1.5), image.shape)
    psf_gauss.run()

    rl = RichardsonLucy(niter=40)
    deconv_image = rl.run(image, psf_gauss.psf_)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    ref_file = os.path.join(dir_path, 'deconv_rl_celegans.tif')
    # imsave(ref_file, deconv_image)
    ref_image = imread(ref_file)

    assert np.allclose(deconv_image, ref_image, rtol=1e-03, atol=1e-03)


def test_richardson_lucy_3d():

    image = data.pollen_poison_noise_blurred()
    psf = data.pollen_psf()

    wiener = RichardsonLucy(niter=40)
    deconv_image = wiener.run(image, psf)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    ref_file = os.path.join(dir_path, 'deconv_rl_pollen.tif')
    # imsave(ref_file, deconv_image)
    ref_image = imread(ref_file)

    assert np.allclose(deconv_image, ref_image, rtol=1e-03, atol=1e-03)
