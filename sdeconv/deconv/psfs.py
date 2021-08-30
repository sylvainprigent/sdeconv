import numpy as np
import math
from .wrappers._psfs import py_gibson_lanni_psf


class PSFGaussian:
    """Generate a Gaussian PSF

    Parameters
    ----------
    sigma: tuple
        Radius of the Gaussian in each dimension
    shape: tuple
        Size of the PSF array in each dimension
    """

    def __init__(self, sigma, shape):
        self.sigma = sigma
        self.shape = shape
        self.psf_ = None

    def run(self):
        """Calculate the PSF image"""
        if len(self.shape) == 2:
            self.psf_ = np.zeros(self.shape)
            x0 = self.shape[0] / 2
            y0 = self.shape[1] / 2
            sigma_x2 = 0.5 / (self.sigma[0] * self.sigma[0])
            sigma_y2 = 0.5 / (self.sigma[1] * self.sigma[1])
            for x in range(self.shape[0]):
                for y in range(self.shape[1]):
                    self.psf_[x, y] = math.exp(- pow(x-x0, 2) * sigma_x2
                                               - pow(y-y0, 2) * sigma_y2)
        elif len(self.shape) == 3:
            x0 = self.shape[0] / 2
            y0 = self.shape[1] / 2
            z0 = self.shape[2] / 2
            sigma_x2 = 0.5 / self.sigma[0] * self.sigma[0]
            sigma_y2 = 0.5 / self.sigma[1] * self.sigma[1]
            sigma_z2 = 0.5 / self.sigma[2] * self.sigma[2]
            for x in range(self.shape[0]):
                for y in range(self.shape[1]):
                    for z in range(self.shape[2]):
                        self.psf_[z, x, y] = math.exp(- pow(x-x0, 2) * sigma_x2
                                                      - pow(y-y0, 2) * sigma_y2
                                                      - pow(z-z0, 2) * sigma_z2)
        else:
            raise Exception('PSFGaussian: can generate only 2D or 3D PSFs')
        return self.psf_


class PSFGibsonLanni:
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

    """

    def __init__(self, shape, res_lateral=100, res_axial=250,
                 numerical_aperture=1.4, lambd=610,
                 ti0=150, ni=1.5, ns=1.33):
        self.shape = shape
        self.res_lateral = res_lateral
        self.res_axial = res_axial
        self.numerical_aperture = numerical_aperture
        self.lambd = lambd
        self.ti0 = ti0
        self.ni = ni
        self.ns = ns
        self.psf_ = None

    def run(self):
        """Calculate the PSF image"""

        self.psf_ = py_gibson_lanni_psf(self.shape[2], self.shape[1],
                                        self.shape[0],
                                        self.res_lateral, self.res_axial,
                                        self.numerical_aperture, self.lambd,
                                        self.ti0, self.ni, self.ns)
        self.psf_ = np.transpose(self.psf_, (2, 0, 1))
        return self.psf_
