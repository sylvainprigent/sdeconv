import numpy as np
import torch
from .interface import SDeconvFilter
from .utils import pad_2d, pad_3d
from sdeconv.core import SSettings


def laplacian_2d(shape):
    image = torch.zeros(shape).to(SSettings.instance().device)

    xc = int(shape[0] / 2)
    yc = int(shape[1] / 2)

    image[xc, yc] = 4
    image[xc, yc - 1] = -1
    image[xc, yc + 1] = -1
    image[xc - 1, yc] = -1
    image[xc + 1, yc] = -1
    return image


def laplacian_3d(shape):
    image = torch.zeros(shape).to(SSettings.instance().device)

    xc = int(shape[2] / 2)
    yc = int(shape[1] / 2)
    zc = int(shape[0] / 2)

    image[zc, yc, xc] = 6
    image[zc - 1, yc, xc] = -1
    image[zc + 1, yc, xc] = -1
    image[zc, yc - 1, xc] = -1
    image[zc, yc + 1, xc] = -1
    image[zc, yc, xc - 1] = -1
    image[zc, yc, xc - 1] = -1
    return image


class SWiener(SDeconvFilter):
    """Apply a gaussian filter

    Parameters
    ----------
    psf: Tensor
        Point spread function
    beta: float
        Regularisation weight
    pad: int/tuple
        Padding in each dimension

    """
    def __init__(self, psf, beta=1e-5, pad=0):
        super().__init__()
        self.psf = psf
        self.beta = beta
        self.pad = pad

    def __call__(self, image):
        if image.ndim == 2:
            image_pad, psf_pad, padding = pad_2d(image, self.psf/torch.sum(self.psf), self.pad)

            fft_source = torch.fft.fft2(image_pad)
            psf_roll = torch.roll(psf_pad, int(-psf_pad.shape[0] / 2), dims=0)
            psf_roll = torch.roll(psf_roll, int(-psf_pad.shape[1] / 2), dims=1)
            fft_psf = torch.fft.fft2(psf_roll)
            fft_laplacian = torch.fft.fft2(laplacian_2d(image_pad.shape))

            den = fft_psf*torch.conj(fft_psf)+self.beta*fft_laplacian*torch.conj(fft_laplacian)
            out_image = torch.real(torch.fft.ifftn((fft_source * torch.conj(fft_psf))/den))
            if image_pad.shape != image.shape:
                return out_image[padding[0]: -padding[0], padding[1]: -padding[1]]
            else:
                return out_image

        elif image.ndim == 3:
            image_pad, psf_pad, padding = pad_3d(image, self.psf/torch.sum(self.psf), self.pad)

            fft_source = torch.fft.fftn(image_pad)
            psf_roll = torch.roll(psf_pad, int(-psf_pad.shape[0] / 2), dims=0)
            psf_roll = torch.roll(psf_roll, int(-psf_pad.shape[1] / 2), dims=1)
            psf_roll = torch.roll(psf_roll, int(-psf_pad.shape[2] / 2), dims=2)

            fft_psf = torch.fft.fftn(psf_roll)
            fft_laplacian = torch.fft.fftn(laplacian_3d(image_pad.shape))

            den = fft_psf*torch.conj(fft_psf)+self.beta*fft_laplacian*torch.conj(fft_laplacian)
            out_image = torch.real(torch.fft.ifftn((fft_source * torch.conj(fft_psf))/den))
            if image_pad.shape != image.shape:
                return out_image[padding[0]:-padding[0],
                                 padding[1]:-padding[1],
                                 padding[2]:-padding[2]]
            else:
                return out_image


metadata = {
    'name': 'SWiener',
    'label': 'Wiener',
    'class': SWiener,
    'parameters': {
        'psf': {
            'type': torch.Tensor,
            'label': 'psf',
            'help': 'Point Spread Function',
            'default': None
        },
        'beta': {
            'type': float,
            'label': 'Beta',
            'help': 'Regularisation parameter',
            'default': 1e-5,
            'range': (0, 999999)
        }
    }
}
