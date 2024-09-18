"""Implements the Gaussian Point Spread Function generator"""
import math
import torch
from sdeconv.core import SSettings
from .interface import SPSFGenerator


class SPSFGaussian(SPSFGenerator):
    """Generate a Gaussian PSF

    :param sigma: width of the PSF. [Z, Y, X] in 3D, [Y, X] in 2D
    :param shape: Shape of the PSF support image. [Z, Y, X] in 3D, [Y, X] in 2D
    """
    def __init__(self,
                 sigma: tuple[float, float] | tuple[float, float, float],
                 shape: tuple[int, int] | tuple[int, int, int]):
        super().__init__()
        self.sigma = sigma
        self.shape = shape
        self.psf_ = None

    @staticmethod
    def _normalize_inputs(sigma: tuple[float, float] | tuple[float, float, float],
                          shape: tuple[int, int, int] | tuple[int, int, int]
                          ) -> tuple:
        """Remove batch dimention if it exists
        
        :param sigma: Width of the PSF
        :param shape: Shape of the PSF
        :return: The modified sigma and shape
        """
        if len(shape) == 3 and shape[0] == 1:
            return sigma[1:], shape[1:]
        return sigma, shape

    def __call__(self) -> torch.Tensor:
        """Calculate the PSF image
        
        :return: The PSF image in a Tensor
        """
        self.sigma, self.shape = SPSFGaussian._normalize_inputs(self.sigma, self.shape)
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
            sigma_x2 = 0.5 / (self.sigma[2] * self.sigma[2])
            sigma_y2 = 0.5 / (self.sigma[1] * self.sigma[1])
            sigma_z2 = 0.5 / (self.sigma[0] * self.sigma[0])

            zzz, yyy, xxx = torch.meshgrid(torch.arange(0, self.shape[0]),
                                           torch.arange(0, self.shape[1]),
                                           torch.arange(0, self.shape[2]),
                                           indexing='ij')
            self.psf_ = torch.exp(- torch.pow(xxx - x_0, 2) * sigma_x2
                                  - torch.pow(yyy - y_0, 2) * sigma_y2
                                  - torch.pow(zzz - z_0, 2) * sigma_z2)

            self.psf_ = self.psf_ / torch.sum(self.psf_)
        else:
            raise ValueError('PSFGaussian: can generate only 2D or 3D PSFs')
        return self.psf_


def spsf_gaussian(sigma: tuple[float, float] | tuple[float, float, float],
                  shape: tuple[int, int] | tuple[int, int, int]
                  ) -> torch.Tensor:
    """Function to generate a Gaussian PSF

    :param sigma: width of the PSF. [Z, Y, X] in 3D, [Y, X] in 2D,
    :param shape: Shape of the PSF support image. [Z, Y, X] in 3D, [Y, X] in 2D,
    :return: The PSF image
    """
    filter_ = SPSFGaussian(sigma, shape)
    return filter_()


metadata = {
    'name': 'SPSFGaussian',
    'label': 'Gaussian PSF',
    'fnc': spsf_gaussian,
    'inputs': {
        'sigma': {
            'type': 'zyx_float',
            'label': 'Sigma',
            'help': 'Gaussian standard deviation in each direction',
            'default': [0, 1.5, 1.5]
        },
        'shape': {
            'type': 'zyx_int',
            'label': 'Size',
            'help': 'PSF image shape',
            'default': [1, 128, 128]
        }
    },
    'outputs': {
        'image': {
            'type': 'Image',
            'label': 'PSF Gaussian'
        },
    }
}
