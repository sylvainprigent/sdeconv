import torch
from sdeconv.core import SSettings


def resize_psf_2d(image, psf):
    kernel = torch.zeros(image.shape).to(SSettings.instance().device)
    x_start = int(image.shape[0] / 2 - psf.shape[0] / 2) + 1
    y_start = int(image.shape[1] / 2 - psf.shape[1] / 2) + 1
    kernel[x_start:x_start + psf.shape[0], y_start:y_start + psf.shape[1]] = psf
    return kernel


def resize_psf_3d(image, psf):
    kernel = torch.zeros(image.shape).to(SSettings.instance().device)
    x_start = int(image.shape[0] / 2 - psf.shape[0] / 2) + 1
    y_start = int(image.shape[1] / 2 - psf.shape[1] / 2) + 1
    z_start = int(image.shape[2] / 2 - psf.shape[2] / 2) + 1
    kernel[x_start:x_start + psf.shape[0], y_start:y_start + psf.shape[1],
           z_start:z_start + psf.shape[2]] = psf
    return kernel


def pad_2d(image, psf, pad):
    """Pad an image and it PSF for deconvolution

    Parameters
    ----------
    image: tensor
        2D image tensor
    psf: tensor
        2D Point Spread Function.
    pad: int/tuple
        Padding in each dimension.

    Returns
    -------
    image, psf, padding: padded versions of the image and the PSF, plus the padding tuple

    """
    padding = pad
    if type(pad) is tuple and len(pad) != image.ndim:
        raise Exception("Padding must be the same dimension as image")
    elif type(pad) is int:
        if pad == 0:
            return image, psf, (0, 0)
        padding = (pad, pad)

    if padding[0] > 0 and padding[1] > 0:

        pad_fn = torch.nn.ReflectionPad2d((padding[0], padding[0], padding[1], padding[1]))
        image_pad = pad_fn(image.detach().clone().to(SSettings.instance().device).view(1, 1, image.shape[0],
                                                       image.shape[1])).view(
            (image.shape[0] + 2 * padding[0], image.shape[1] + 2 * padding[0]))
    else:
        image_pad = image.detach().clone().to(SSettings.instance().device)
    psf_pad = resize_psf_2d(image_pad, psf)
    return image_pad, psf_pad, padding


def pad_3d(image, psf, pad):
    """Pad an image and it PSF for deconvolution

    Parameters
    ----------
    image: tensor
        2D image tensor
    psf: tensor
        2D Point Spread Function.
    pad: int/tuple
        Padding in each dimension.

    Returns
    -------
    image, psf, padding: padded versions of the image and the PSF, plus the padding tuple

    """
    padding = pad
    if type(pad) is tuple and len(pad) != image.ndim:
        raise Exception("Padding must be the same dimension as image")
    elif type(pad) is int:
        if pad == 0:
            return image, psf, (0, 0, 0)
        padding = (pad, pad, pad)

    if padding[0] > 0 and padding[1] > 0 and padding[2] > 0:
        p3d = (padding[2], padding[2], padding[1], padding[1], padding[0], padding[0])
        pad_fn = torch.nn.ReflectionPad3d(p3d)
        image_pad = pad_fn(
            image.detach().clone().to(SSettings.instance().device).view(1, 1, image.shape[0], image.shape[1], image.shape[2])).view(
            (image.shape[0] + 2 * padding[0], image.shape[1] + 2 * padding[1], image.shape[2] + 2 * padding[2]))
        psf_pad = torch.nn.functional.pad(psf, p3d, "constant", 0)
    else:
        image_pad = image
        psf_pad = psf
    return image_pad, psf_pad, padding
