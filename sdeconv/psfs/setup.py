"""Setup the psfs module"""
from numpy.distutils.misc_util import Configuration


def configuration(parent_package='', top_path=None):
    """Submodule configuration

    Parameters
    ----------
    parent_package: str
        Name of the parent package
    top_path: str
        Path of the top module

    """
    config = Configuration('psfs', parent_package, top_path)
    config.add_subpackage('wrappers')
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
