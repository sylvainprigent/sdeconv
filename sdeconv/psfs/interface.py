"""Interface for a psf generator

Classes
-------
SPSFGenerator

"""

from sdeconv.core import SObservable


class SPSFGenerator(SObservable):
    """Interface for a psf generator"""
    def __init__(self):
        super().__init__()
        self.type = 'SPSFGenerator'

    def __call__(self):
        """Generate the PSF

        Return
        ------
        Tensor: PSF image [(Z) Y X]

        """
        raise NotImplementedError('SPSFGenerator is an interface. Please implement the'
                                  ' __call__ method')
