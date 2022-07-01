"""Implements the Gaussian Point Spread Function generator"""
import math
import torch
from sdeconv.core import SSettings
from .interface import SPSFGenerator


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
            x_0 = math.floor(self.shape[0] / 2)
            y_0 = math.floor(self.shape[1] / 2)
            # print('center= (', x0, ', ', y0, ')')
            sigma_x2 = 0.5 / (self.sigma[0] * self.sigma[0])
            sigma_y2 = 0.5 / (self.sigma[1] * self.sigma[1])

            xx_, yy_ = torch.meshgrid(torch.arange(0, self.shape[0]),
                                      torch.arange(0, self.shape[1]),
                                      indexing='ij')
            self.psf_ = torch.exp(- torch.pow(xx_ - x_0, 2) * sigma_x2
                                  - torch.pow(yy_ - y_0, 2) * sigma_y2)
            self.psf_ = self.psf_ / torch.sum(self.psf_)
        elif len(self.shape) == 3:
            self.psf_ = torch.zeros(self.shape).to(SSettings.instance().device)
            x_0 = math.floor(self.shape[2] / 2)
            y_0 = math.floor(self.shape[1] / 2)
            z_0 = math.floor(self.shape[0] / 2)
            sigma_x2 = 0.5 / self.sigma[2] * self.sigma[2]
            sigma_y2 = 0.5 / self.sigma[1] * self.sigma[1]
            sigma_z2 = 0.5 / self.sigma[0] * self.sigma[0]

            zzz, yyy, xxx = torch.meshgrid(torch.arange(0, self.shape[0]),
                                           torch.arange(0, self.shape[1]),
                                           torch.arange(0, self.shape[2]),
                                           indexing='ij')
            self.psf_ = torch.exp(- torch.pow(xxx - x_0, 2) * sigma_x2
                                  - torch.pow(yyy - y_0, 2) * sigma_y2
                                  - torch.pow(zzz - z_0, 2) * sigma_z2)

            self.psf_ = self.psf_ / torch.sum(self.psf_)
        else:
            raise Exception('PSFGaussian: can generate only 2D or 3D PSFs')
        return self.psf_


metadata = {
    'name': 'SPSFGaussian',
    'label': 'Gaussian PSF',
    'class': SPSFGaussian,
    'parameters': {
        'sigma': {
            'type': 'zyx',
            'label': 'Sigma',
            'help': 'Gaussian standard deviation in each direction',
            'default': [1.5, 1.5, 0]
        },
        'shape': {
            'type': 'zyx',
            'label': 'Size',
            'help': 'Regularisation parameter',
            'default': [128, 128, 1]
        }
    }
}