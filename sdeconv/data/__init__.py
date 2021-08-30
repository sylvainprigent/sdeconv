import os.path as osp
import os

__all__ = ['celegans',
           'pollen',
           'pollen_poison_noise_blurred',
           'pollen_psf']

legacy_data_dir = osp.abspath(osp.dirname(__file__))


def _fetch(data_filename):
    """Fetch a given data file from the local data dir.

    This function provides the path location of the data file given
    its name in the scikit-image repository.

    Parameters
    ----------
    data_filename:
        Name of the file in the scikit-bioimaging data dir

    Returns
    -------
    Path of the local file as a python string.
    """

    filepath = os.path.join(legacy_data_dir, data_filename)

    if os.path.isfile(filepath):
        return filepath
    else:
        raise FileExistsError("Cannot find the file:", filepath)    


def _load(f):
    """Load an image file located in the data directory.
    Parameters
    ----------
    f : string
        File name.
    Returns
    -------
    img : ndarray
        Image loaded from ``simglibpy.data_dir``.
    """
    # importing io is quite slow since it scans all the backends
    # we lazy import it here
    from skimage.io import imread
    return imread(_fetch(f))


def celegans():
    """2D confocal (Airyscan) image of a c. elegans intestine.

    Returns
    -------
    pollen : (316, 316) uint16 ndarray
        2D confocal image of a C. elegans intestine.
    """

    return _load("celegans.tif")


def pollen():
    """3D Pollen image.

    Returns
    -------
    pollen : (32, 256, 256) uint16 ndarray
        Pollen image.
    """

    return _load("pollen.tif")


def pollen_poison_noise_blurred():
    """3D Pollen image corrupted with Poisson noise and blurred .

    Returns
    -------
    pollen : (32, 256, 256) uint16 ndarray
        Corrupted pollen image.
    """

    return _load("pollen_poisson_noise_blurred.tif")


def pollen_psf():
    """3D PSF to deblur the pollen image.

    Returns
    -------
    pollen : (32, 256, 256) uint16 ndarray
        Corrupted pollen image.
    """

    return _load("pollen_psf.tif")
