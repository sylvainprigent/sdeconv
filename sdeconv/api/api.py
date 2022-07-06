"""Application programing interface for SDeconv

Classes
-------
SDeconvAPI

"""

import os
import importlib
import torch
from .factory import SDeconvModuleFactory, SDeconvFactoryError


class SDeconvAPI:
    """Main API to call SDeconv methods.
    The API implements a factory that instantiate the deconvolution
    """
    def __init__(self):
        self.psfs = SDeconvModuleFactory()
        self.filters = SDeconvModuleFactory()
        for name in self._find_modules('deconv'):
            mod = importlib.import_module(name)
            self.filters.register(mod.metadata['name'], mod.metadata)
        for name in self._find_modules('psfs'):
            mod = importlib.import_module(name)
            self.psfs.register(mod.metadata['name'], mod.metadata)

    @staticmethod
    def _find_modules(directory):
        path = os.path.abspath(os.path.dirname(__file__))
        path = os.path.dirname(path)
        modules = []
        for parent in [directory]:
            path_ = os.path.join(path, parent)
            for module_path in os.listdir(path_):
                if module_path.endswith(
                        ".py") and 'setup' not in module_path and \
                        'interface' not in module_path and \
                        '__init__' not in module_path and not module_path.startswith(
                        "_"):
                    modules.append(f"sdeconv.{parent}.{module_path.split('.')[0]}")
        return modules

    def filter(self, name, **kwargs):
        """Instantiate a deconvolution filter

        Parameters
        ----------
        name: str
            Unique name of the filter to instantiate
        kwargs: dict
            arguments of the filter

        """
        if name == 'None':
            return None
        return self.filters.get(name, **kwargs)

    def psf(self, method_name, **kwargs):
        """Instantiate a psf generator

        Parameters
        ----------
        method_name: str
            name of the PSF generator to instantiate
        kwargs: dict
            parameters of the PSF generator

        """
        if method_name == 'None':
            return None
        filter_ = self.psfs.get(method_name, **kwargs)
        if filter_.type == 'SPSFGenerator':
            return filter_
        raise SDeconvFactoryError(f'The method {method_name} is not a PSF generator')

    def generate_psf(self, method_name, **kwargs):
        """Generates a Point SPread Function

        Parameters
        ----------
        method_name: str
            Name of the PSF Generator method
        kwargs: dict
            Parameters of the PSF generator

        """
        # print('generate psf args=', **kwargs)
        generator = self.psf(method_name, **kwargs)
        return generator()

    def deconvolve(self, image, method_name, plane_by_plane, **kwargs):
        """Run the deconvolution on an image

        Parameters
        ----------
        image: torch.Tensor
            Image to deconvolve. Can be 2D to 5D
        method_name: str
            Name of the deconvolution method to use
        plane_by_plane: bool
            True to process the image plane by plane when dimension is more than 2
        kwargs: dict
            Parameters of the deconvolution method

        """
        filter_ = self.filter(method_name, **kwargs)
        if filter_.type == 'SDeconvFilter':
            return self._deconv_dims(image, filter_, plane_by_plane=plane_by_plane)
        raise SDeconvFactoryError(f'The method {method_name} is not a deconvolution filter')

    @staticmethod
    def _deconv_3d_by_plane(image, filter_):
        """Call the 3D deconvolution plane by plane

        Parameters
        ----------
        image: torch.Tensor
            3D image tensor
        filter_: class
            deconvolution class

        """
        out_image = torch.zeros(image.shape)
        for i in range(image.shape[0]):
            out_image[i, ...] = filter_(image[i, ...])
        return out_image

    @staticmethod
    def _deconv_4d(image, filter_):
        """Call the 3D+t deconvolution

        Parameters
        ----------
        image: torch.Tensor
            3D+t image tensor
        filter_: class
            deconvolution class

        """
        out_image = torch.zeros(image.shape)
        for i in range(image.shape[0]):
            out_image[i, ...] = filter_(image[i, ...])
        return out_image

    @staticmethod
    def _deconv_4d_by_plane(image, filter_):
        """Call the 3D+t deconvolution plane by plane

        Parameters
        ----------
        image: torch.Tensor
            3D+t image tensor
        filter_: class
            deconvolution class

        """
        out_image = torch.zeros(image.shape)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                out_image[i, j, ...] = filter_(image[i, j, ...])
        return out_image

    @staticmethod
    def _deconv_5d(image, filter_):
        """Call the 3D+t multi-channel deconvolution

        Parameters
        ----------
        image: torch.Tensor
            3D+t multi-channel image tensor
        filter_: class
            deconvolution class

        """
        out_image = torch.zeros(image.shape)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                out_image[i, j, ...] = filter_(image[i, j, ...])
        return out_image

    @staticmethod
    def _deconv_5d_by_plane(image, filter_):
        """Call the 3D+t multi-channel deconvolution plane by plane

        Parameters
        ----------
        image: torch.Tensor
            3D+t multi-channel image tensor
        filter_: class
            deconvolution class

        """
        out_image = torch.zeros(image.shape)
        for batch in range(image.shape[0]):
            for channel in range(image.shape[1]):
                for plane in range(image.shape[2]):
                    out_image[batch, channel, plane, ...] = \
                        filter_(image[batch, channel, plane, ...])
        return out_image

    @staticmethod
    def _deconv_dims(image, filter_, plane_by_plane=False):
        """Call the deconvolution method depending on the image dimension

        Parameters
        ----------
        image: torch.Tensor
            3D+t multi-channel image tensor
        filter_: class
            deconvolution class

        """
        out_image = None
        if image.ndim == 2:
            out_image = filter_(image)
        elif image.ndim == 3 and plane_by_plane:
            out_image = SDeconvAPI._deconv_3d_by_plane(image, filter_)
        elif image.ndim == 3 and not plane_by_plane:
            out_image = filter_(image)
        elif image.ndim == 4 and not plane_by_plane:
            out_image = SDeconvAPI._deconv_4d(image, filter_)
        elif image.ndim == 4 and plane_by_plane:
            out_image = SDeconvAPI._deconv_4d_by_plane(image, filter_)
        elif image.ndim == 5 and not plane_by_plane:
            out_image = SDeconvAPI._deconv_5d(image, filter_)
        elif image.ndim == 5 and plane_by_plane:
            out_image = SDeconvAPI._deconv_5d_by_plane(image, filter_)
        else:
            raise SDeconvFactoryError('SDeconv can process only images up to 5 dims')
        return out_image
