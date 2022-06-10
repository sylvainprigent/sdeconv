import torch
import torch.nn.functional as F
from .interface import SDeconvFilter
from .utils import pad_2d, pad_3d


class SRichardsonLucy(SDeconvFilter):
    """Implements the Richardson-Lucy deconvolution

    Parameters
    ----------
    psf: Tensor
        Point spread function

    """
    def __init__(self, psf, niter=30, pad=0):
        super().__init__()
        self.psf = psf
        self.niter = niter
        self.pad = pad

    @staticmethod
    def _resize_psf(psf, width, height):
        kernel = torch.zeros((width, height))
        x_start = int(width / 2 - psf.shape[0] / 2) + 1
        y_start = int(height / 2 - psf.shape[1] / 2) + 1
        kernel[x_start:x_start+psf.shape[0], y_start:y_start+psf.shape[1]] = psf
        return kernel

    def __call__(self, image):
        if image.ndim == 2:
            image_pad, psf_pad, padding = pad_2d(image, self.psf/torch.sum(self.psf), self.pad)

            psf_roll = torch.roll(psf_pad, [int(-psf_pad.shape[0]/2),
                                            int(-psf_pad.shape[1]/2)], dims=(0, 1))
            fft_psf = torch.fft.fft2(psf_roll)
            fft_psf_mirror = torch.fft.fft2(torch.flip(psf_roll, dims=[0, 1]))

            out_image = image_pad.detach().clone()
            for _ in range(self.niter):
                fft_out = torch.fft.fft2(out_image)
                fft_tmp = fft_out*fft_psf
                tmp = torch.real(torch.fft.ifft2(fft_tmp))
                tmp = image_pad/tmp
                fft_tmp = torch.fft.fft2(tmp)
                fft_tmp = fft_tmp * fft_psf_mirror
                tmp = torch.real(torch.fft.ifft2(fft_tmp))
                out_image = out_image * tmp

            if image_pad.shape != image.shape:
                return out_image[padding[0]:-padding[0], padding[1]:-padding[1]]
            else:
                return out_image

        elif image.ndim == 3:
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
            else:
                return out_image


metadata = {
    'name': 'SRichardsonLucy',
    'label': 'Richardson-Lucy',
    'class': SRichardsonLucy,
    'parameters': {
        'psf': {
            'type': torch.Tensor,
            'label': 'psf',
            'help': 'Point Spread Function',
            'default': None
        },
        'niter': {
            'type': int,
            'label': 'niter',
            'help': 'Number of iterations',
            'default': 30,
            'range': (0, 999999)
        }
    }
}
