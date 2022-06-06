import math
import torch
from .interface import SDeconvFilter
from .wiener import SWiener


def hv_loss(img, weighting):
    """Sparse Hessian regularization term

    Parameters
    ----------
    img: Tensor
        Tensor of shape BCYX containing the estimated image
    weighting: float
        Sparse weighting parameter in [0, 1]. 0 sparse, and 1 not sparse

    """
    a, b, h, w = img.size()
    dxx2 = torch.square(-img[:, :, 2:, 1:-1] + 2 * img[:, :, 1:-1, 1:-1] - img[:, :, :-2, 1:-1])
    dyy2 = torch.square(-img[:, :, 1:-1, 2:] + 2 * img[:, :, 1:-1, 1:-1] - img[:, :, 1:-1, :-2])
    dxy2 = torch.square(img[:, :, 2:, 2:] - img[:, :, 2:, 1:-1] - img[:, :, 1:-1, 2:] +
                        img[:, :, 1:-1, 1:-1])
    hv = torch.sqrt(weighting * weighting * (dxx2 + dyy2 + 2 * dxy2) +
                    (1 - weighting) * (1 - weighting) * torch.square(img[:, :, 1:-1, 1:-1])).sum()
    return hv / (a * b * h * w)


def hv_loss_3d(img, delta, weighting):
    """Sparse Hessian regularization term

    Parameters
    ----------
    img: Tensor
        Tensor of shape BCZYX containing the estimated image
    delta: float
        Resolution delta between XY and Z
    weighting: float
        Sparse weighting parameter in [0, 1]. 0 sparse, and 1 not sparse

    """
    img_ = img[:, 1:-1, 1:-1, 1:-1]
    d11 = img[:, 1:-1, 1:-1, 2:] - 2*img_ - img[:, 1:-1, 1:-1, :-2]
    d22 = img[:, 1:-1, 2:, 1:-1] - 2*img_ - img[:, 1:-1, :-2, 1:-1]
    d33 = delta*delta*(img[:, 2:, 1:-1, 1:-1] - 2*img_ - img[:, :-2, 1:-1, 1:-1])
    d12_d21 = img[:, 1:-1, 2:, 2:] - img[:, 1:-1, 1:-1, 2:] - img[:, 1:-1, 2:, 1:-1] + img_
    d13_d31 = delta*(img[:, 2:, 1:-1, 2:] - img[:, 1:-1, 1:-1, 2:] - img[:,  2:, 1:-1, 1:-1] + img_)
    d23_d32 = delta*(img[:, 2:, 2:, 1:-1] - img[:, 1:-1, 2:, 1:-1] - img[:,  2:, 1:-1, 1:-1] + img_)
    d00 = img_

    hv = torch.square(weighting*d11) + torch.square(weighting*d22) + torch.square(
        weighting*d33) + 2 * torch.square(weighting*d12_d21) + 2 * torch.square(
        weighting*d13_d31) + 2 * torch.square(weighting*d23_d32) + torch.square((1-weighting)*d00)

    return torch.mean(torch.sqrt(hv))


def dataterm_deconv(blurry_image, deblurred_image, psf):
    """Deconvolution L2 data-term

    Compute the L2 error between the original image and the convoluted reconstructed image

    Parameters
    ----------
    blurry_image: Tensor
        Tensor of shape BCYX containing the original blurry image
    deblurred_image: Tensor
        Tensor of shape BCYX containing the estimated deblurred image
    psf: Tensor
        Tensor containing the point spread function

    """
    conv_op = torch.nn.Conv2d(1, 1, kernel_size=psf.shape[2],
                              stride=1,
                              padding=int((psf.shape[2] - 1) / 2),
                              bias=False)
    with torch.no_grad():
        conv_op.weight = torch.nn.Parameter(psf)
    mse = torch.nn.MSELoss()
    return mse(blurry_image, conv_op(deblurred_image))


def dataterm_deconv_3d(blurry_image, deblurred_image, fft_psf):
    """Deconvolution L2 data-term

    Compute the L2 error between the original image and the convoluted reconstructed image

    Parameters
    ----------
    blurry_image: Tensor
        Tensor of shape BCZYX containing the original blurry image
    deblurred_image: Tensor
        Tensor of shape BCZYX containing the estimated deblurred image
    fft_psf: Tensor
        Tensor containing the point spread function Fourier transform

    """
    fft_deblurred_image = torch.fft.fftn(deblurred_image)
    conv_deblured_image = torch.real(torch.fft.ifftn(fft_deblurred_image*fft_psf))

    mse = torch.nn.MSELoss()
    return mse(blurry_image, conv_deblured_image)


class Spitfire(SDeconvFilter):
    """Gray scaled image deconvolution with the Spitfire algorithm

    Parameters
    ----------
    psf: Tensor
        Point spread function
    weight: float
        model weight between hessian and sparsity. Value is in  ]0, 1[
    reg: float
        Regularization weight. Value is in [0, 1]

    """
    def __init__(self, psf, weight=0.6, reg=0.995, gradient_step=0.01, precision=1e-7):
        super().__init__()
        self.psf = psf
        self.weight = weight
        self.reg = reg
        self.precision = precision
        self.niter_ = 0
        self.max_iter_ = 2000
        self.gradient_step_ = gradient_step
        self.loss_ = None

    def __call__(self, image):
        if image.ndim == 2:
            return self.run_2d(image)
        elif image.ndim == 3:
            return self.run_3d(image)

    def run_2d(self, image):
        self.progress(0)
        mini = torch.min(image)
        maxi = torch.max(image)
        image = (image-mini)/(maxi-mini)
        # pad image
        padding = 13
        pad_fn = torch.nn.ReflectionPad2d(padding)
        image_pad = pad_fn(image.detach().clone().view(1, 1, image.shape[0], image.shape[1]))

        self.psf = self.psf.view(1, 1, self.psf.shape[0], self.psf.shape[1])
        deconv_image = image_pad.detach().clone()
        deconv_image.requires_grad = True
        optimizer = torch.optim.Adam([deconv_image], lr=self.gradient_step_)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        previous_loss = 9e12
        count_eq = 0
        self.niter_ = 0
        for i in range(self.max_iter_):
            self.progress(int(100*i/self.max_iter_))
            self.niter_ += 1
            optimizer.zero_grad()
            loss = self.reg * dataterm_deconv(image_pad, deconv_image, self.psf) + \
                (1-self.reg) * hv_loss(deconv_image, self.weight)
            print('iter:', self.niter_, ' loss:', loss.item())
            if abs(loss - previous_loss) < self.precision:
                count_eq += 1
            else:
                previous_loss = loss
                count_eq = 0
            if count_eq > 5:
                break
            loss.backward()
            optimizer.step()
            scheduler.step()
        self.loss_ = loss
        self.progress(100)
        deconv_image = deconv_image.view(image_pad.shape[2],
                                         image_pad.shape[3])[padding:-padding, padding:-padding]
        return (maxi-mini)*deconv_image + mini

    def run_3d(self, image):
        delta = 1
        self.progress(0)
        mini = torch.min(image)
        maxi = torch.max(image)
        image = (image-mini)/(maxi-mini)

        #w_filter = SWiener(self.psf, beta=1e-5)
        #deconv_image = w_filter(image)
        #deconv_image = (deconv_image - torch.min(deconv_image))/(torch.max(deconv_image)-torch.min(deconv_image))
        deconv_image = torch.ones(image.shape) # image.detach().clone()

        image = image.view(1, image.shape[0], image.shape[1], image.shape[2])
        deconv_image = deconv_image.view(1, deconv_image.shape[0], deconv_image.shape[1], deconv_image.shape[2])
        deconv_image.requires_grad = True
        optimizer = torch.optim.Adam([deconv_image], lr=self.gradient_step_)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        previous_loss = 9e12
        count_eq = 0
        self.niter_ = 0

        psf_roll = torch.roll(self.psf, int(-self.psf.shape[0] / 2), dims=0)
        psf_roll = torch.roll(psf_roll, int(-self.psf.shape[1] / 2), dims=1)
        psf_roll = torch.roll(psf_roll, int(-self.psf.shape[2] / 2), dims=2)
        psf_roll.view(1, self.psf.shape[0], self.psf.shape[1], self.psf.shape[2])
        fft_psf = torch.fft.fftn(psf_roll)
        for i in range(self.max_iter_):
            self.progress(int(100*i/self.max_iter_))
            self.niter_ += 1
            optimizer.zero_grad()
            # loss = self.reg * dataterm_deconv_3d(image, deconv_image, fft_psf) + \
            #     (1-self.reg) * hv_loss_3d(deconv_image, delta, self.weight)
            loss = dataterm_deconv_3d(image, deconv_image, fft_psf)
            print('iter:', self.niter_, ' loss:', loss.item())
            if abs(loss - previous_loss) < self.precision:
                count_eq += 1
            else:
                previous_loss = loss
                count_eq = 0
            if count_eq > 5:
                break
            loss.backward()
            optimizer.step()
            scheduler.step()
        self.loss_ = loss
        self.progress(100)
        deconv_image = deconv_image.view(image.shape[1],
                                         image.shape[2],
                                         image.shape[3])
        return deconv_image
        # return (maxi-mini)*deconv_image + mini


metadata = {
    'name': 'Spitfire',
    'label': 'Spitfire',
    'class': Spitfire,
    'parameters': {
        'psf': {
            'type': torch.Tensor,
            'label': 'psf',
            'help': 'Point Spread Function',
            'default': None
        },
        'weight': {
            'type': float,
            'label': 'weight',
            'help': 'Model weight between hessian and sparsity. Value is in  ]0, 1[',
            'default': 0.6,
            'range': (0, 1)
        },
        'reg': {
            'type': float,
            'label': 'Regularization',
            'help': 'Regularization weight. Value is in [0, 1]',
            'default': 0.995,
            'range': (0, 1)
        }
    }
}
