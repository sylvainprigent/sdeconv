"""Interface for a psf generator"""
import torch
from sdeconv.core import SObservable


class SPSFGenerator(SObservable):
    """Interface for a psf generator"""
    def __init__(self):
        super().__init__()
        self.type = 'SPSFGenerator'

    def __call__(self) -> torch.Tensor:
        """Generate the PSF

        return: PSF image [(Z) Y X]
        """
        raise NotImplementedError('SPSFGenerator is an interface. Please implement the'
                                  ' __call__ method')
