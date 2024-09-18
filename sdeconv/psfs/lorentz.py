"""Implements the Lorentz Point Spread Function generator"""
import math
import torch
from ..core import SSettings
from .interface import SPSFGenerator


class SPSFLorentz(SPSFGenerator):
    """Generate a Lorentz PSF

    :param gamma: Width of the Lorentz in each dimension [(Z), Y, X]
    :param shape: Size of the PSF image in each dimension [(Z), Y, X]
    """
    def __init__(self,
                 gamma: tuple[float, float] | tuple[float, float, float],
                 shape: tuple[int, int, int] | tuple[int, int, int]):
        super().__init__()
        self.gamma = gamma
        self.shape = shape
        self.psf_ = None

    @staticmethod
    def _normalize_inputs(gamma: tuple[float, float] | tuple[float, float, float],
                          shape: tuple[int, int, int] | tuple[int, int, int]
                          ) -> tuple:
        """Remove batch dimention if it exists
        
        :param sigma: Width of the PSF
        :param shape: Shape of the PSF
        :return: The modified sigma and shape
        """
        if len(shape) == 3 and shape[0] == 1:
            return gamma[1:], shape[1:]
        return gamma, shape

    def __call__(self) -> torch.Tensor:
        """Calculate the PSF image
        
        :return: The PSF image as a Tensor
        """
        self.gamma, self.shape = SPSFLorentz._normalize_inputs(self.gamma, self.shape)
        if len(self.shape) == 2:
            self.psf_ = torch.zeros((self.shape[0], self.shape[1])).to(SSettings.instance().device)
            x_0 = math.floor(self.shape[0] / 2)
            y_0 = math.floor(self.shape[1] / 2)
            # print('center= (', x0, ', ', y0, ')')
            xx_, yy_ = torch.meshgrid(torch.arange(0, self.shape[0]),
                                      torch.arange(0, self.shape[1]),
                                      indexing='ij')
            self.psf_ = 1 / (1 + torch.pow((xx_ - x_0)/(0.5*self.gamma[0]), 2) +
                             torch.pow((yy_ - y_0)/(0.5*self.gamma[1]), 2))
            self.psf_ = self.psf_ / torch.sum(self.psf_)
        elif len(self.shape) == 3:
            self.psf_ = torch.zeros(self.shape).to(SSettings.instance().device)
            x_0 = math.floor(self.shape[2] / 2)
            y_0 = math.floor(self.shape[1] / 2)
            z_0 = math.floor(self.shape[0] / 2)
            zzz, yyy, xxx = torch.meshgrid(torch.arange(0, self.shape[0]),
                                           torch.arange(0, self.shape[1]),
                                           torch.arange(0, self.shape[2]),
                                           indexing='ij')
            self.psf_ = 1 / (1 + torch.pow((xxx - x_0)/(0.5*self.gamma[2]), 2) +
                             torch.pow((yyy - y_0)/(0.5*self.gamma[1]), 2) +
                             torch.pow((zzz - z_0)/(0.5*self.gamma[0]), 2))
            self.psf_ = self.psf_ / torch.sum(self.psf_)
        else:
            raise ValueError('SPSFLorentz: can generate only 2D or 3D PSFs')
        return self.psf_


def spsf_lorentz(gamma: tuple[float, float] | tuple[float, float, float],
                 shape: tuple[int, int, int] | tuple[int, int, int]
                 ) -> torch.Tensor:
    """Function to generate a Lorentz PSF

    :param gamma: Width of the Lorentz in each dimension [(Z), Y, X],
    :param shape: Size of the PSF image in each dimension [(Z), Y, X],
    :return: The PSF mage
    """
    filter_ = SPSFLorentz(gamma, shape)
    return filter_()


metadata = {
    'name': 'SPSFLorentz',
    'label': 'Lorentz PSF',
    'fnc': spsf_lorentz,
    'inputs': {
        'gamma': {
            'type': 'zyx_float',
            'label': 'Gamma',
            'help': 'PSF width in each direction',
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
            'label': 'PSF Lorentz'
        },
    }
}
