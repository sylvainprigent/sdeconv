"""Module that provides sample images"""

import os.path as osp
import os
import numpy as np
import torch
from skimage.io import imread
from sdeconv.core import SSettings


__all__ = ['celegans',
           'pollen',
           'pollen_poison_noise_blurred',
           'pollen_psf']

legacy_data_dir = osp.abspath(osp.dirname(__file__))


def _fetch(data_filename: str) -> str:
    """Fetch a given data file from the local data dir.

    This function provides the path location of the data file given
    its name in the scikit-image repository.

    :param data_filename: Name of the file in the scikit-bioimaging data dir,
    :return: Path of the local file as a python string
    """

    filepath = os.path.join(legacy_data_dir, data_filename)

    if os.path.isfile(filepath):
        return filepath
    raise FileExistsError("Cannot find the file:", filepath)


def _load(filename: str) -> np.ndarray:
    """Load an image file located in the data directory.
    
    :param: filename: Path of the file to load.
    :return: The data loaded in a numpy array
    """
    return torch.tensor(np.float32(imread(_fetch(filename)))).to(SSettings.instance().device)


def celegans():
    """2D confocal (Airyscan) image of a c. elegans intestine.

    :return: (310, 310) uint16 ndarray
    """
    return _load("celegans.tif")[3:-3, 3:-3]


def pollen():
    """3D Pollen image.

    :return: (32, 256, 256) uint16 ndarray
    """
    return _load("pollen.tif")


def pollen_poison_noise_blurred():
    """3D Pollen image corrupted with Poisson noise and blurred .

    :return: (32, 256, 256) uint16 ndarray
    """
    return _load("pollen_poisson_noise_blurred.tif")


def pollen_psf():
    """3D PSF to deblur the pollen image.

    :return: (32, 256, 256) uint16 ndarray
    """
    psf = _load("pollen_psf.tif")
    return psf / torch.sum(psf)
