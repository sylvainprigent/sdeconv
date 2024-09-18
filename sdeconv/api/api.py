"""Application programing interface for SDeconv"""

import os
import importlib
import torch

from ..psfs import SPSFGenerator
from ..deconv import SDeconvFilter
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
    def _find_modules(directory: str) -> list[str]:
        """Search sub modules in a directory
        
        :param directory: Directory to search
        :return: The founded module names 
        """
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

    def filter(self, name: str, **kwargs) -> SDeconvFilter | None:
        """Instantiate a deconvolution filter

        :param name: Unique name of the filter to instantiate
        :param kwargs: arguments of the filter
        :return: An instance of the filter
        """
        if name == 'None':
            return None
        return self.filters.get(name, **kwargs)

    def psf(self, method_name, **kwargs) -> SPSFGenerator:
        """Instantiate a psf generator

        :param method_name: name of the PSF generator
        :param kwargs: parameters of the PSF generator
        :return: An instance of the generator
        """
        if method_name == 'None':
            return None
        filter_ = self.psfs.get(method_name, **kwargs)
        if filter_.type == 'SPSFGenerator':
            return filter_
        raise SDeconvFactoryError(f'The method {method_name} is not a PSF generator')

    def generate_psf(self, method_name, **kwargs) -> torch.Tensor:
        """Generates a Point SPread Function

        :param method_name: Name of the PSF Generator method
        :param kwargs: Parameters of the PSF generator
        :return: The generated PSF
        """
        # print('generate psf args=', **kwargs)
        generator = self.psf(method_name, **kwargs)
        return generator()

    def deconvolve(self,
                   image: torch.Tensor,
                   method_name: str,
                   plane_by_plane: bool,
                   **kwargs
                   ) -> torch.Tensor:
        """Run the deconvolution on an image

        :param image: Image to deconvolve. Can be 2D to 5D
        :param method_name: Name of the deconvolution method to use
        :param plane_by_plane: True to process the image plane by plane 
                               when dimension is more than 2
        :param kwargs: Parameters of the deconvolution method
        :return: The deblurred image
        """
        filter_ = self.filter(method_name, **kwargs)
        if filter_.type == 'SDeconvFilter':
            return self._deconv_dims(image, filter_, plane_by_plane=plane_by_plane)
        raise SDeconvFactoryError(f'The method {method_name} is not a deconvolution filter')

    @staticmethod
    def _deconv_3d_by_plane(image: torch.Tensor, filter_: SDeconvFilter) -> torch.Tensor:
        """Call the 3D deconvolution plane by plane

        :param image: 3D image tensor
        :param filter_: deconvolution class
        :return: The deblurred image
        """
        out_image = torch.zeros(image.shape)
        for i in range(image.shape[0]):
            out_image[i, ...] = filter_(image[i, ...])
        return out_image

    @staticmethod
    def _deconv_4d(image: torch.Tensor, filter_: SDeconvFilter) -> torch.Tensor:
        """Call the 3D+t deconvolution

        :param image: 3D+t image tensor
        :param filter_: deconvolution class
        :return: The deblurred stack
        """
        out_image = torch.zeros(image.shape)
        for i in range(image.shape[0]):
            out_image[i, ...] = filter_(image[i, ...])
        return out_image

    @staticmethod
    def _deconv_4d_by_plane(image: torch.Tensor, filter_: SDeconvFilter) -> torch.Tensor:
        """Call the 3D+t deconvolution plane by plane

        :param image: 3D+t image tensor
        :param filter_: deconvolution class
        :return: The deblurred stack
        """
        out_image = torch.zeros(image.shape)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                out_image[i, j, ...] = filter_(image[i, j, ...])
        return out_image

    @staticmethod
    def _deconv_5d(image: torch.Tensor, filter_: SDeconvFilter) -> torch.Tensor:
        """Call the 3D+t multi-channel deconvolution

        :param image: 3D+t image tensor
        :param filter_: deconvolution class
        :return: The deblurred hyper-stack
        """
        out_image = torch.zeros(image.shape)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                out_image[i, j, ...] = filter_(image[i, j, ...])
        return out_image

    @staticmethod
    def _deconv_5d_by_plane(image: torch.Tensor, filter_: SDeconvFilter) -> torch.Tensor:
        """Call the 3D+t multi-channel deconvolution plane by plane

        :param image: 3D+t image tensor
        :param filter_: deconvolution class
        :return: The deblurred hyper-stack
        """
        out_image = torch.zeros(image.shape)
        for batch in range(image.shape[0]):
            for channel in range(image.shape[1]):
                for plane in range(image.shape[2]):
                    out_image[batch, channel, plane, ...] = \
                        filter_(image[batch, channel, plane, ...])
        return out_image

    @staticmethod
    def _deconv_dims(image: torch.Tensor,
                     filter_: SDeconvFilter,
                     plane_by_plane: bool = False):
        """Call the deconvolution method depending on the image dimension

        :param image: 3D+t image tensor
        :param filter_: deconvolution class
        :param plane_by_plane: True to deblur third dimention as independent planes
        :return: The deblurred image, stack or hyper-stack
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
