import os
import numpy as np
from skimage.io import imread, imsave
import torch

from sdeconv.data import celegans, pollen_poison_noise_blurred, pollen_psf
from sdeconv.deconv import SWiener
from sdeconv.psfs import SPSFGaussian


# tmp_path is a pytest fixture
def test_wiener_2d(tmp_path):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    image = torch.Tensor(celegans())

    psf_generator = SPSFGaussian((1.5, 1.5), (13, 13))
    psf = psf_generator()

    filter_ = SWiener(psf, beta=1.5)
    out_image = filter_(image)

    # imsave(os.path.join(root_dir, 'celegans_wiener.tif'), out_image.detach().numpy())
    ref_image = imread(os.path.join(root_dir, 'celegans_wiener.tif'))

    np.testing.assert_equal(out_image.detach().numpy(), ref_image)


def test_wiener_3d(tmp_path):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    image = torch.Tensor(np.float32(pollen_poison_noise_blurred()))
    psf = torch.Tensor(np.float32(pollen_psf()))

    filter_ = SWiener(psf, beta=8)
    out_image = filter_(image)

    # imsave(os.path.join(root_dir, 'pollen_wiener.tif'), out_image.detach().numpy())
    ref_image = imread(os.path.join(root_dir, 'pollen_wiener.tif'))

    np.testing.assert_equal(out_image.detach().numpy(), ref_image)
