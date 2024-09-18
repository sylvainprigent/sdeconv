"""Implements the Wiener deconvolution for 2D and 3D images"""
import torch
from sdeconv.core import SSettings
from .interface import SDeconvFilter
from ._utils import pad_2d, pad_3d, unpad_3d, np_to_torch


def laplacian_2d(shape: tuple[int, int]) -> torch.Tensor:
    """Define the 2D laplacian matrix

    :param shape: 2D image shape
    :return: torch.Tensor with size defined by shape and laplacian coefficients at it center
    """
    image = torch.zeros(shape).to(SSettings.instance().device)

    x_c = int(shape[0] / 2)
    y_c = int(shape[1] / 2)

    image[x_c, y_c] = 4
    image[x_c, y_c - 1] = -1
    image[x_c, y_c + 1] = -1
    image[x_c - 1, y_c] = -1
    image[x_c + 1, y_c] = -1
    return image


def laplacian_3d(shape: tuple[int, int]) -> torch.Tensor:
    """Define the 3D laplacian matrix

    :param shape: 3D image shape
    :return: torch.Tensor with size defined by shape and laplacian coefficients at it center
    """
    image = torch.zeros(shape).to(SSettings.instance().device)

    x_c = int(shape[2] / 2)
    y_c = int(shape[1] / 2)
    z_c = int(shape[0] / 2)

    image[z_c, y_c, x_c] = 6
    image[z_c - 1, y_c, x_c] = -1
    image[z_c + 1, y_c, x_c] = -1
    image[z_c, y_c - 1, x_c] = -1
    image[z_c, y_c + 1, x_c] = -1
    image[z_c, y_c, x_c - 1] = -1
    image[z_c, y_c, x_c - 1] = -1
    return image


class SWiener(SDeconvFilter):
    """Apply a Wiener deconvolution

    :param psf: Point spread function
    :param beta: Regularisation weight
    :param pad: Padding in each dimension
    """
    def __init__(self,
                 psf: torch.Tensor,
                 beta: float = 1e-5,
                 pad: int | tuple[int, int] | tuple[int, int, int] = 0):
        super().__init__()
        self.psf = psf
        self.beta = beta
        self.pad = pad

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Run the Wiener deconvolution
        
        :param image: Blurry image for a single channel time point [(Z) Y X]
        :return: deblurred image [(Z) Y X]
        """
        if image.ndim == 2:
            return self._wiener_2d(image)
        if image.ndim == 3:
            return self._wiener_3d(image)
        raise ValueError('Wiener can only deblur 2D or 3D tensors')

    def _wiener_2d(self, image: torch.Tensor) -> torch.Tensor:
        """Compute the 2D wiener deconvolution

        :param image: 2D image tensor
        :return: torch.Tensor of the 2D deblurred image
        """
        image_pad, psf_pad, padding = pad_2d(image, self.psf / torch.sum(self.psf), self.pad)

        fft_source = torch.fft.fft2(image_pad)
        psf_roll = torch.roll(psf_pad, int(-psf_pad.shape[0] / 2), dims=0)
        psf_roll = torch.roll(psf_roll, int(-psf_pad.shape[1] / 2), dims=1)
        fft_psf = torch.fft.fft2(psf_roll)
        fft_laplacian = torch.fft.fft2(laplacian_2d(image_pad.shape))

        den = fft_psf * torch.conj(fft_psf) + self.beta * fft_laplacian * torch.conj(fft_laplacian)
        out_image = torch.real(torch.fft.ifftn((fft_source * torch.conj(fft_psf)) / den))
        if image_pad.shape != image.shape:
            return out_image[padding[0]: -padding[0], padding[1]: -padding[1]]
        return out_image

    def _wiener_3d(self, image: torch.Tensor) -> torch.Tensor:
        """Compute the 3D wiener deconvolution

        :param image: 2D image tensor
        :return: torch.Tensor of the 2D deblurred image
        """
        image_pad, psf_pad, padding = pad_3d(image, self.psf / torch.sum(self.psf), self.pad)

        fft_source = torch.fft.fftn(image_pad)
        psf_roll = torch.roll(psf_pad, int(-psf_pad.shape[0] / 2), dims=0)
        psf_roll = torch.roll(psf_roll, int(-psf_pad.shape[1] / 2), dims=1)
        psf_roll = torch.roll(psf_roll, int(-psf_pad.shape[2] / 2), dims=2)

        fft_psf = torch.fft.fftn(psf_roll)
        fft_laplacian = torch.fft.fftn(laplacian_3d(image_pad.shape))

        den = fft_psf * torch.conj(fft_psf) + self.beta * fft_laplacian * torch.conj(fft_laplacian)
        out_image = torch.real(torch.fft.ifftn((fft_source * torch.conj(fft_psf)) / den))
        if image_pad.shape != image.shape:
            return unpad_3d(out_image, padding)
        return out_image


def swiener(image: torch.Tensor,
            psf: torch.Tensor,
            beta: float = 1e-5,
            pad: int | tuple[int, int] | tuple[int, int, int] = 0
            ):
    """Convenient function to call the SWiener on numpy array

    :param image: Image to deblur,
    :param psf: Point spread function,
    :param beta: Regularisation weight,
    :param pad: Padding in each dimension,
    :return: the deblurred image
    """
    psf_ = np_to_torch(psf)
    filter_ = SWiener(psf_, beta, pad)
    return filter_(np_to_torch(image))


metadata = {
    'name': 'SWiener',
    'label': 'Wiener',
    'fnc': swiener,
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
        'beta': {
            'type': 'float',
            'label': 'Beta',
            'help': 'Regularisation parameter',
            'default': 1e-5,
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
            'label': 'Wiener'
        },
    }
}
