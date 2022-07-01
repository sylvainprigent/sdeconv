"""Interface for a deconvolution filter

Classes
-------
SDeconvFilter

"""

from sdeconv.core import SObservable


class SDeconvFilter(SObservable):
    """Interface for a deconvolution filter"""
    def __init__(self):
        super().__init__()
        self.type = 'SDeconvFilter'

    def __call__(self, image):
        """Do the deconvolution

        Parameters
        ----------
        image: Tensor
            Blurry image for a single channel time point [(Z) Y X]

        Return
        ------
        Tensor: deblurred image [(Z) Y X]

        """
        raise NotImplementedError('SDeconvFilter is an interface. Please implement the'
                                  ' __call__ method')
