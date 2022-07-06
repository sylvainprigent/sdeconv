"""Module to test the Gibson-Lanni PSF implementation"""
import os
import numpy as np
from skimage.io import imread
from sdeconv.psfs.gibson_lanni import SPSFGibsonLanni


def test_gibson_lanni():
    """An example of how you might test your plugin."""

    shape = (18, 128, 128)
    NA = 1.4
    wavelength = 0.610
    M = 100
    ns = 1.33
    ng0 = 1.5
    ng = 1.5
    ni0 = 1.5
    ni = 1.5
    ti0 = 150
    tg0 = 170
    tg = 170
    res_lateral = 0.1
    res_axial = 0.25
    pZ = 0
    use_square = False

    psf_generator = SPSFGibsonLanni(shape, NA, wavelength, M, ns,
                                    ng0, ng, ni0, ni, ti0, tg0, tg,
                                    res_lateral, res_axial, pZ, use_square)
    psf = psf_generator()

    root_dir = os.path.dirname(os.path.abspath(__file__))
    # imsave(os.path.join(root_dir, 'gibsonlanni.tif'), psf.detach().numpy())
    ref_psf = imread(os.path.join(root_dir, 'gibsonlanni.tif'))

    np.testing.assert_almost_equal(psf.detach().cpu().numpy(), ref_psf, decimal=5)
