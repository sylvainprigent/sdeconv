import os
import numpy as np
from skimage.io import imread, imsave
import torch

from sdeconv.data import celegans, pollen_poison_noise_blurred, pollen_psf
from sdeconv.deconv import SRichardsonLucy
from sdeconv.psfs import SPSFGaussian


# tmp_path is a pytest fixture
def test_richardson_lucy_2d(tmp_path):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    image = celegans()

    psf_generator = SPSFGaussian((1.5, 1.5), (13, 13))
    psf = psf_generator()

    filter_ = SRichardsonLucy(psf, niter=30, pad=13)
    out_image = filter_(image)

    # imsave(os.path.join(root_dir, 'celegans_richardson_lucy_gpu.tif'), out_image.detach().cpu().numpy())
    ref_image = imread(os.path.join(root_dir, 'celegans_richardson_lucy.tif'))

    np.testing.assert_almost_equal(out_image.detach().cpu().numpy(), ref_image, decimal=1)


def test_richardson_lucy_3d(tmp_path):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    image = pollen_poison_noise_blurred()
    psf = pollen_psf()

    filter_ = SRichardsonLucy(psf, niter=30, pad=(16, 64, 64))
    out_image = filter_(image)

    # imsave(os.path.join(root_dir, 'pollen_richardson_lucy_linux.tif'), out_image.detach().cpu().numpy())
    ref_image = imread(os.path.join(root_dir, 'pollen_richardson_lucy.tif'))

    np.testing.assert_almost_equal(out_image[20, ...].detach().cpu().numpy(), ref_image[20, ...],
                                   decimal=1)
