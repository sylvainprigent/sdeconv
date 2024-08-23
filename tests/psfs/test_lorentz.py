"""Unit testing for Lorentz PSFs"""

import os
import numpy as np
from skimage.io import imread, imsave

from sdeconv.psfs.lorentz import SPSFLorentz


def test_psf_lorentz_2d():
    """Unit testing the 2D Lorentz PSF generator"""

    psf_generator = SPSFLorentz((1.5, 1.5), (15, 15))
    psf = psf_generator()

    root_dir = os.path.dirname(os.path.abspath(__file__))
    #imsave(os.path.join(root_dir, 'lorentz2d.tif'), psf.detach().numpy())
    ref_psf = imread(os.path.join(root_dir, 'lorentz2d.tif'))

    np.testing.assert_almost_equal(psf.detach().cpu().numpy(), ref_psf, decimal=5)


def test_psf_lorentz_3d():
    """Unit testing the 3D Lorentz PSF generator"""

    psf_generator = SPSFLorentz((0.5, 1.5, 1.5), (11, 15, 15))
    psf = psf_generator()

    root_dir = os.path.dirname(os.path.abspath(__file__))
    #imsave(os.path.join(root_dir, 'lorentz3d.tif'), psf.detach().numpy())
    ref_psf = imread(os.path.join(root_dir, 'lorentz3d.tif'))

    np.testing.assert_almost_equal(psf.detach().cpu().numpy(), ref_psf, decimal=5)
