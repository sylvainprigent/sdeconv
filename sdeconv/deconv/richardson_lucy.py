"""Richardson Lucy deconvolution

Classes
-------
RichardsonLucy

"""

import math
import numpy as np
from .wrappers._richardson_lucy_deconv import (py_richardson_lucy_deconv_2d,
                                               py_richardson_lucy_deconv_3d)


class RichardsonLucy:
    """Deconvolution of an image using the Richardson-Lucy algorithm

    Parameters
    ----------
    niter: int
        Maximum number of iterations
    """

    def __init__(self, niter: int = 40):
        self.iter = niter
        self.deconvolved_ = None

    def run(self, image: np.array, psf: np.array):
        im = image.astype(np.float32)
        imin = np.amin(im)
        imax = np.amax(im)
        psf = psf.astype(np.float32)
        psf_ = psf / np.sum(psf)
        im = im / math.sqrt(np.sum(np.square(im)))
        if im.ndim == 2:
            self.deconvolved_ = py_richardson_lucy_deconv_2d(im, psf_,
                                                             self.iter)
            self.deconvolved_ = self.deconvolved_ * (imax - imin) + imin
        elif im.ndim == 3:
            self.deconvolved_ = py_richardson_lucy_deconv_3d(im, psf_,
                                                             self.iter)
            self.deconvolved_ = self.deconvolved_ * (imax - imin) + imin
        return self.deconvolved_
