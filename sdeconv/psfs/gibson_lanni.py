from .wrappers.gibsonlanni import py_gibson_lanni_psf

import torch

from .interface import SPSFGenerator
from sdeconv.core import SSettings
import numpy as np


class SPSFGibsonLanni(SPSFGenerator):
    """Generate a Gibson-Lanni PSF

    Parameters
    ----------
    shape: tuple
        Size of the PSF array in each dimension
    res_lateral: float
        Lateral resolution
    res_axial: float
        Axial resolution
    numerical_aperture: float
        Numerical aperture
    lambd: float
        Illumination wavelength
    ti0: float
        Working distance
    ni: float
        Refractive index immersion
    ns: float
        Refractive index sample
    use_square: bool
        If true, calculate the square of the Gibson-Lanni model to simulate a pinhole. It then gives
        a PSF for a confocal image
    """

    def __init__(self, shape, res_lateral=100, res_axial=250,
                 numerical_aperture=1.4, lambd=610,
                 ti0=150, ni=1.5, ns=1.33, use_square=False):
        super().__init__()
        self.shape = shape
        self.res_lateral = res_lateral
        self.res_axial = res_axial
        self.numerical_aperture = numerical_aperture
        self.lambd = lambd
        self.ti0 = ti0
        self.ni = ni
        self.ns = ns
        self.use_square = use_square
        self.psf_ = None

    def __call__(self):
        """Calculate the PSF image"""

        self.psf_ = py_gibson_lanni_psf(self.shape[2], self.shape[1],
                                        self.shape[0],
                                        self.res_lateral, self.res_axial,
                                        self.numerical_aperture, self.lambd,
                                        self.ti0, self.ni, self.ns)
        self.psf_ = torch.tensor(np.transpose(self.psf_, (2, 0, 1))).to(SSettings.instance().device)
        if self.use_square:
            self.psf_ = torch.square(self.psf_)
        return self.psf_
