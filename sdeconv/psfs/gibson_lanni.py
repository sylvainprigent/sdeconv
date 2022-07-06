"""Implementation of Gibson Lanni Point Spread Function model

Classes
-------
SPSFGibsonLanni

"""
import numpy as np
import torch

from sdeconv.core import SSettings
from .interface import SPSFGenerator
from .wrappers.gibsonlanni import py_gibson_lanni_psf


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
                 ti0=150, n_i=1.5, n_s=1.33, use_square=False):
        super().__init__()
        self.shape = shape
        self.res_lateral = res_lateral
        self.res_axial = res_axial
        self.numerical_aperture = numerical_aperture
        self.lambd = lambd
        self.ti0 = ti0
        self.n_i = n_i
        self.n_s = n_s
        self.use_square = use_square
        self.psf_ = None

    def __call__(self):
        """Calculate the PSF image"""
        self.psf_ = py_gibson_lanni_psf(self.shape[2], self.shape[1],
                                        self.shape[0],
                                        self.res_lateral, self.res_axial,
                                        self.numerical_aperture, self.lambd,
                                        self.ti0, self.n_i, self.n_s)
        self.psf_ = torch.tensor(np.transpose(self.psf_, (2, 0, 1))).to(SSettings.instance().device)
        if self.use_square:
            self.psf_ = torch.square(self.psf_)
        return self.psf_


def spsf_gibson_lanni(shape, res_lateral=100, res_axial=250,
                      numerical_aperture=1.4, lambd=610,
                      ti0=150, n_i=1.5, n_s=1.33, use_square=False):
    filter_ = SPSFGibsonLanni(shape, res_lateral, res_axial,
                              numerical_aperture, lambd,
                              ti0, n_i, n_s, use_square)
    return filter_()


metadata = {
    'name': 'SPSFGibsonLanni',
    'label': 'Gibson Lanni PSF',
    'fnc': spsf_gibson_lanni,
    'inputs': {
        'shape': {
            'type': 'zyx_int',
            'label': 'Size',
            'help': 'Regularisation parameter',
            'default': [11, 128, 128]
        },
        'res_lateral': {
            'type': 'float',
            'label': 'Lateral resolution',
            'help': 'Lateral resolution',
            'default': 100
        },
        'res_axial': {
            'type': 'float',
            'label': 'Axial resolution',
            'help': 'Axial resolution',
            'default': 250
        },
        'numerical_aperture': {
            'type': 'float',
            'label': 'Numerical aperture',
            'help': 'Numerical aperture',
            'default': 1.4
        },
        'lambd': {
            'type': 'float',
            'label': 'Illumination wavelength',
            'help': 'Illumination wavelength',
            'default': 610
        },
        'ti0': {
            'type': 'float',
            'label': 'Working distance',
            'help': 'Working distance',
            'default': 150
        },
        'n_i': {
            'type': 'float',
            'label': 'Refractive index immersion',
            'help': 'Refractive index immersion',
            'default': 1.5
        },
        'n_s': {
            'type': 'float',
            'label': 'Refractive index sample',
            'help': 'Refractive index sample',
            'default': 1.33
        },
        'use_square': {
            'type': 'bool',
            'label': 'Confocal',
            'help': 'Check for confocal PSF, uncheck for widefield',
            'default': True
        }
    },
    'outputs': {
        'image': {
            'type': 'Image',
            'label': 'PSF Gibson-Lanni'
        },
    }
}
