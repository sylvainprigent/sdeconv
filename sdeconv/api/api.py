import os
import importlib
from .factory import SDeconvModuleFactory


class SDeconvAPI:
    def __init__(self):
        self.filters = SDeconvModuleFactory()
        discovered_modules = self._find_modules()
        for name in discovered_modules:
            # print('register the module:', name)
            mod = importlib.import_module(name)
            # print(mod.__name__)
            self.filters.register(mod.metadata['name'], mod.metadata)

    @staticmethod
    def _find_modules():
        path = os.path.abspath(os.path.dirname(__file__))
        path = os.path.dirname(path)
        modules = []
        for parent in ['enhancing', 'reconstruction', 'registration']:
            path_ = os.path.join(path, parent)
            for x in os.listdir(path_):
                if x.endswith(
                        ".py") and 'interface' not in x and '__init__' not in x and not x.startswith(
                        "_"):
                    modules.append(f"sairyscan.{parent}.{x.split('.')[0]}")
        return modules

    def filter(self, name, **args):
        if name == 'None':
            return None
        return self.filters.get(name, **args)

    def psf(self, method_name, **args):
        filter_ = self.filter(method_name, **args)
        # TODO: check if the filter is instance of PSF interface
        return filter_()

    def deconvolve(self, image, method_name, **args):
        filter_ = self.filter(method_name, **args)
        # TODO: check if the filter is instance of SDeconvInterface
        return filter_(image)
