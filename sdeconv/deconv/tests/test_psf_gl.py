import pytest

import os
import numpy as np
from sdeconv import data
from sdeconv.deconv import PSFGibsonLanni
from skimage.io import imread, imsave


def test_psf_gl():

    shape = (18, 128, 128)
    res_lateral = 100
    res_axial = 250
    numerical_aperture = 1.4
    lambd = 610
    ti0 = 150
    ni = 1.5
    ns = 1.33

    obj_psf = PSFGibsonLanni(shape, res_lateral, res_axial,
                             numerical_aperture, lambd,
                             ti0, ni, ns)
    psf = obj_psf.run()                         

    dir_path = os.path.dirname(os.path.realpath(__file__))
    ref_file = os.path.join(dir_path, 'psf_gl.tif')
    # imsave(ref_file, deconv_image)
    ref_image = imread(ref_file)

    print('ref image shape', ref_image.shape)
    print('test image shape', psf.shape)

    assert np.array_equal(psf, ref_image)

