"""Unit testing for Gaussian PSFs"""

import os
import numpy as np
from skimage.io import imread

from sdeconv.psfs.gaussian import SPSFGaussian


def test_psf_gaussian_2d():
    """Unit testing the 2D Gaussian PSF generator"""

    psf_generator = SPSFGaussian((1.5, 1.5), (15, 15))
    psf = psf_generator()

    root_dir = os.path.dirname(os.path.abspath(__file__))
    # imsave(os.path.join(root_dir, 'gaussian2d.tif'), psf.detach().numpy())
    ref_psf = imread(os.path.join(root_dir, 'gaussian2d.tif'))

    np.testing.assert_almost_equal(psf.detach().cpu().numpy(), ref_psf, decimal=5)


def test_psf_gaussian_3d():
    """Unit testing the 3D Gaussian PSF generator"""

    psf_generator = SPSFGaussian((0.5, 1.5, 1.5), (11, 15, 15))
    psf = psf_generator()

    root_dir = os.path.dirname(os.path.abspath(__file__))
    # imsave(os.path.join(root_dir, 'gaussian3d.tif'), psf.detach().numpy())
    ref_psf = imread(os.path.join(root_dir, 'gaussian3d.tif'))

    np.testing.assert_almost_equal(psf.detach().cpu().numpy(), ref_psf, decimal=5)
