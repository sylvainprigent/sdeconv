"""Implementation of Richardson-Lucy deconvolution for 2D and 3D images"""
import torch
import numpy as np
from .interface import SDeconvFilter
from ._utils import pad_2d, pad_3d, np_to_torch


class SRichardsonLucy(SDeconvFilter):
    """Implements the Richardson-Lucy deconvolution

    :param psf: Point spread function
    :param niter: Number of iterations
    :param pad: image padding size
    """
    def __init__(self,
                 psf: torch.Tensor,
                 niter: int = 30,
                 pad: int | tuple[int, int] | tuple[int, int, int] = 0):
        super().__init__()
        self.psf = psf
        self.niter = niter
        self.pad = pad

    @staticmethod
    def _resize_psf(psf, width, height) -> torch.Tensor:
        """Resize the PSF to match the image size for Fourier transform

        :param psf: Point spread function
        :param width: Width of the resized PSF
        :param height: Height of the resized PSF
        :return: The resized PSF
        """
        kernel = torch.zeros((width, height))
        x_start = int(width / 2 - psf.shape[0] / 2) + 1
        y_start = int(height / 2 - psf.shape[1] / 2) + 1
        kernel[x_start:x_start+psf.shape[0], y_start:y_start+psf.shape[1]] = psf
        return kernel

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Apply the Richardson-Lucy deconvolution
        
        :param image: Blurry image for a single channel time point [(Z) Y X]
        :return: deblurred image [(Z) Y X]
        """
        if image.ndim == 2:
            return self._deconv_2d(image)
        if image.ndim == 3:
            return self._deconv_3d(image)
        raise ValueError('Richardson-Lucy can only deblur 2D or 3D tensors')

    def _deconv_2d(self, image: torch.Tensor) -> torch.Tensor:
        """Implements Richardson-Lucy for 2D images

        :param image: 2D image tensor
        :return: 2D deblurred image
        """
        image_pad, psf_pad, padding = pad_2d(image, self.psf / torch.sum(self.psf), self.pad)

        psf_roll = torch.roll(psf_pad, [int(-psf_pad.shape[0] / 2),
                                        int(-psf_pad.shape[1] / 2)], dims=(0, 1))
        fft_psf = torch.fft.fft2(psf_roll)
        fft_psf_mirror = torch.fft.fft2(torch.flip(psf_roll, dims=[0, 1]))

        out_image = image_pad.detach().clone()
        for _ in range(self.niter):
            fft_out = torch.fft.fft2(out_image)
            fft_tmp = fft_out * fft_psf
            tmp = torch.real(torch.fft.ifft2(fft_tmp))
            tmp = image_pad / tmp
            fft_tmp = torch.fft.fft2(tmp)
            fft_tmp = fft_tmp * fft_psf_mirror
            tmp = torch.real(torch.fft.ifft2(fft_tmp))
            out_image = out_image * tmp

        if image_pad.shape != image.shape:
            return out_image[padding[0]:-padding[0], padding[1]:-padding[1]]
        return out_image

    def _deconv_3d(self, image: torch.Tensor) -> torch.Tensor:
        """Implements Richardson-Lucy for 3D images

        :param image: 3D image tensor
        :return: 3D deblurred image
        """
        image_pad, psf_pad, padding = pad_3d(image, self.psf / torch.sum(self.psf), self.pad)

        psf_roll = torch.roll(psf_pad, int(-psf_pad.shape[0] / 2), dims=0)
        psf_roll = torch.roll(psf_roll, int(-psf_pad.shape[1] / 2), dims=1)
        psf_roll = torch.roll(psf_roll, int(-psf_pad.shape[2] / 2), dims=2)

        fft_psf = torch.fft.fftn(psf_roll)
        fft_psf_mirror = torch.fft.fftn(torch.flip(psf_roll, dims=[0, 1]))

        out_image = image_pad.detach().clone()
        for _ in range(self.niter):
            fft_out = torch.fft.fftn(out_image)
            fft_tmp = fft_out * fft_psf
            tmp = torch.real(torch.fft.ifftn(fft_tmp))
            tmp = image_pad / tmp
            fft_tmp = torch.fft.fftn(tmp)
            fft_tmp = fft_tmp * fft_psf_mirror
            tmp = torch.real(torch.fft.ifftn(fft_tmp))
            out_image = out_image * tmp

        if image_pad.shape != image.shape:
            return out_image[padding[0]:-padding[0],
                             padding[1]:-padding[1],
                             padding[2]:-padding[2]]
        return out_image


def srichardsonlucy(image: torch.Tensor,
                    psf: torch.Tensor,
                    niter: int = 30,
                    pad: int | tuple[int, int] | tuple[int, int, int] = 0
                    ) -> torch.Tensor:
    """Convenient function to call the SRichardsonLucy using numpy array

    :param image: Image to deblur
    :param psf: Point spread function
    :param niter: Number of iterations
    :param pad: image padding size
    :return: the deblurred image
    """
    psf_ = np_to_torch(psf)
    image_ = np_to_torch(image)
    filter_ = SRichardsonLucy(psf_, niter, pad)
    if isinstance(image, np.ndarray):
        return filter_(image_)
    return filter_(image)


metadata = {
    'name': 'SRichardsonLucy',
    'label': 'Richardson-Lucy',
    'fnc': srichardsonlucy,
    'inputs': {
        'image': {
            'type': 'Image',
            'label': 'Image',
            'help': 'Input image'
        },
        'psf': {
            'type': 'Image',
            'label': 'PSF',
            'help': 'Point Spread Function'
        },
        'niter': {
            'type': 'int',
            'label': 'niter',
            'help': 'Number of iterations',
            'default': 30,
            'range': (0, 999999)
        },
        'pad': {
            'type': 'int',
            'label': 'Padding',
            'help': 'Padding to avoid spectrum artifacts',
            'default': 13,
            'range': (0, 999999)
        }
    },
    'outputs': {
        'image': {
            'type': 'Image',
            'label': 'Richardson-Lucy'
        },
    }
}
