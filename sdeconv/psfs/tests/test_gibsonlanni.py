import os
import numpy as np
from skimage.io import imread, imsave
import torch

from sdeconv.psfs.gibson_lanni import SPSFGibsonLanni


# tmp_path is a pytest fixture
def test_gibson_lanni(tmp_path):
    """An example of how you might test your plugin."""

    #shape = (18, 128, 128)
    #res_lateral = 100
    #res_axial = 250
    #numerical_aperture = 1.4
    #lambd = 610
    #ti0 = 150
    #ni = 1.5
    #ns = 1.33

    #psf_generator = SPSFGibsonLanni(shape, res_lateral, res_axial,
    #                                numerical_aperture, lambd,
    #                                ti0, ni, ns)
    #psf = psf_generator()

    #root_dir = os.path.dirname(os.path.abspath(__file__))
    ## imsave(os.path.join(root_dir, 'gibsonlanni.tif'), psf.detach().numpy())
    #ref_psf = imread(os.path.join(root_dir, 'gibsonlanni.tif'))

    #np.testing.assert_almost_equal(psf.detach().cpu().numpy(), ref_psf, decimal=5)
    return True
