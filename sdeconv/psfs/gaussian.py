import math
import torch

from .interface import SPSFGenerator
from sdeconv.core import SSettings


class SPSFGaussian(SPSFGenerator):
    """Generate a Gaussian PSF

    Parameters
    ----------
    sigma: tuple
        Radius of the Gaussian in each dimension
    shape: tuple
        Size of the PSF array in each dimension

    """
    def __init__(self, sigma, shape):
        super().__init__()
        self.sigma = sigma
        self.shape = shape
        self.psf_ = None

    def __call__(self):
        """Calculate the PSF image"""
        if len(self.shape) == 2:
            self.psf_ = torch.zeros((self.shape[0], self.shape[1])).to(SSettings.instance().device)
            x0 = math.floor(self.shape[0] / 2)
            y0 = math.floor(self.shape[1] / 2)
            # print('center= (', x0, ', ', y0, ')')
            sigma_x2 = 0.5 / (self.sigma[0] * self.sigma[0])
            sigma_y2 = 0.5 / (self.sigma[1] * self.sigma[1])
            for x in range(self.shape[0]):
                for y in range(self.shape[1]):
                    self.psf_[x, y] = math.exp(- pow(x-x0, 2) * sigma_x2
                                               - pow(y-y0, 2) * sigma_y2)
            self.psf_ = self.psf_ / torch.sum(self.psf_)
        elif len(self.shape) == 3:
            self.psf_ = torch.zeros(self.shape).to(SSettings.instance().device)
            x0 = math.floor(self.shape[2] / 2)
            y0 = math.floor(self.shape[1] / 2)
            z0 = math.floor(self.shape[0] / 2)
            sigma_x2 = 0.5 / self.sigma[2] * self.sigma[2]
            sigma_y2 = 0.5 / self.sigma[1] * self.sigma[1]
            sigma_z2 = 0.5 / self.sigma[0] * self.sigma[0]
            for x in range(self.shape[2]):
                for y in range(self.shape[1]):
                    for z in range(self.shape[0]):
                        self.psf_[z, y, x] = math.exp(- pow(x-x0, 2) * sigma_x2
                                                      - pow(y-y0, 2) * sigma_y2
                                                      - pow(z-z0, 2) * sigma_z2)
            self.psf_ = self.psf_ / torch.sum(self.psf_)
        else:
            raise Exception('PSFGaussian: can generate only 2D or 3D PSFs')
        return self.psf_
