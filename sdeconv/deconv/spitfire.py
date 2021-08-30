"""Sparse and TV/HV denoising classes

Classes
-------
SpitfireDenoise

"""

import math
import numpy as np
from .wrappers._spitfire_deconv import (py_spitfire_deconv_2d,
                                        py_spitfire_deconv_3d)


class SpitfireDeconv:
    """Deconvolution of an image using the sparse variation model

    Parameters
    ----------
    regularization: float
        Regularization parameter. It is express in power of 2
        (reg = pow(2, -regularization))
    weighting: float
        weighting parameter between the Hessian and Intensity term
        [0.6, 0.9, 1.0]
    model: str
        Regularization model (SV or HV)
    niter: int
        Maximum number of iterations
    """

    def __init__(self, regularization: float = 12, weighting: float = 0.6,
                 model: str = 'HV', niter: int = 200, deltaz: float = 1.0,
                 deltat: float = 1.0):
        self.regularization = regularization
        self.weighting = weighting
        self.iter = niter
        self.model = model
        self.deltaz = deltaz
        self.deltat = deltat
        self.deconvolved_ = None

    def run(self, image: np.array, psf: np.array):
        im = image.astype(np.float32)
        imin = np.amin(im)
        imax = np.amax(im)
        psf = psf.astype(np.float32)
        psf_ = psf / np.sum(psf)
        im = im / math.sqrt(np.sum(np.square(im)))
        if im.ndim == 2:
            self.deconvolved_ = py_spitfire_deconv_2d(im, psf_,
                                                      self.regularization,
                                                      self.weighting,
                                                      self.model,
                                                      self.iter)
            self.deconvolved_ = self.deconvolved_ * (imax - imin) + imin
        elif im.ndim == 3:
            self.deconvolved_ = py_spitfire_deconv_3d(im, psf_,
                                                      self.regularization,
                                                      self.weighting,
                                                      self.model,
                                                      self.iter,
                                                      self.deltaz)
            self.deconvolved_ = self.deconvolved_ * (imax - imin) + imin
        return self.deconvolved_
