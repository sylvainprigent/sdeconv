import os
import importlib
from .factory import SDeconvModuleFactory


class SDeconvAPI:
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
            for x in os.listdir(path_):
                if x.endswith(
                        ".py") and 'setup' not in x and 'interface' not in x and '__init__' not in x and not x.startswith(
                        "_"):
                    modules.append(f"sdeconv.{parent}.{x.split('.')[0]}")
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
        else:
            raise SDeconvFactoryError(f'The method {method_name} is not a PSF generator')

    def generate_psf(self, method_name, **kwargs):
        generator = self.psf(method_name, **kwargs)
        return generator()

    def deconvolve(self, image, method_name, plane_by_plane, **args):
        filter_ = self.filter(method_name, **args)
        if filter_.type == 'SDeconvFilter':
            return self._deconv_dims(image, filter_, plane_by_plane=plane_by_plane)
        else:
            raise SDeconvFactoryError(f'The method {method_name} is not a deconvolution filter')

    @staticmethod
    def _deconv_dims(image, filter_, plane_by_plane=False):
        if image.ndim == 2:
            return filter_(image)
        elif image.ndim == 3 and plane_by_plane:
            out_image = torch.zeros(image.shape)
            for p in range(image.shape[0]):
                out_image[p, ...] = filter_(image[p, ...])
            return out_image
        elif image.ndim == 3 and not plane_by_plane:
            return filter_(image)
        elif image.ndim == 4 and not plane_by_plane:
            out_image = torch.zeros(image.shape)
            for b in range(image.shape[0]):
                out_image[b, ...] = filter_(image[b, ...])
            return out_image
        elif image.ndim == 4 and plane_by_plane:
            out_image = torch.zeros(image.shape)
            for b in range(image.shape[0]):
                for z in range(image.shape[1]):
                    out_image[b, z, ...] = filter_(image[b, z, ...])
            return out_image
        elif image.ndim == 5 and not plane_by_plane:
            out_image = torch.zeros(image.shape)
            for b in range(image.shape[0]):
                for c in range(image.shape[1]):
                    out_image[b, c, ...] = filter_(image[b, c, ...])
            return out_image
        elif image.ndim == 5 and plane_by_plane:
            out_image = torch.zeros(image.shape)
            for b in range(image.shape[0]):
                for c in range(image.shape[1]):
                    for z in range(image.shape[2]):
                        out_image[b, c, z, ...] = filter_(image[b, c, z, ...])
            return out_image
        else:
            raise SDeconvFactoryError('SDeconv can process only images up to 5 dims')
