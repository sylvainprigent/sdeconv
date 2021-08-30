"""Wiener deconvolution

Classes
-------
WienerDeconv

"""

import math
import numpy as np
from .wrappers._wiener_deconv import (py_wiener_deconv_2d,
                                      py_wiener_deconv_3d)


class WienerDeconv:
    """Deconvolution of an image using the Richardson-Lucy algorithm

    Parameters
    ----------
    lambda_: float
        Maximum number of iterations
    """

    def __init__(self, lambda_: float = 0.05, connectivity: int = 4):
        self.lambda_ = lambda_
        self.connectivity = connectivity
        self.deconvolved_ = None

    def run(self, image: np.array, psf: np.array):
        im = image.astype(np.float32)
        imin = np.amin(im)
        imax = np.amax(im)
        psf = psf.astype(np.float32)
        psf_ = psf / np.sum(psf)
        im = im / math.sqrt(np.sum(np.square(im)))
        if im.ndim == 2:
            self.deconvolved_ = py_wiener_deconv_2d(im, psf_,
                                                    self.lambda_,
                                                    self.connectivity)
            self.deconvolved_ = self.deconvolved_ * (imax - imin) + imin
        elif im.ndim == 3:
            self.deconvolved_ = py_wiener_deconv_3d(im, psf_,
                                                    self.lambda_,
                                                    self.connectivity)
            self.deconvolved_ = self.deconvolved_ * (imax - imin) + imin
        return self.deconvolved_
