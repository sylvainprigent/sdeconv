"""Interface for a deconvolution filter"""
import torch
from sdeconv.core import SObservable


class SDeconvFilter(SObservable):
    """Interface for a deconvolution filter

    All the algorithm settings must be set in the `__init__` method (PSF included) and the
    `__call__` method is used to actually do the calculation
    """
    def __init__(self):
        super().__init__()
        self.type = 'SDeconvFilter'

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Do the deconvolution

        :param image: Blurry image for a single channel time point [(Z) Y X]
        :return: deblurred image [(Z) Y X]
        """
        raise NotImplementedError('SDeconvFilter is an interface. Please implement the'
                                  ' __call__ method')
