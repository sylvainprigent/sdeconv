"""Implementation of deconvolution using noise2void algorithm"""
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
import torch
from torch.utils.data import DataLoader

from .interface_nn import NNModule
from ._datasets import SelfSupervisedPatchDataset
from ._datasets import SelfSupervisedDataset
from ._transforms import FlipAugmentation, VisionScale


def generate_2d_points(shape: tuple[int, int], n_point: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate random 2D coordinates to mask

    :param shape: Shape of the image to mask
    :param n_point: Number of coordinates to mask
    :return: (y, x) coordinates to mask
    """
    idy_msk = np.random.randint(0, int(shape[0]/2), n_point)
    idx_msk = np.random.randint(0, int(shape[1]/2), n_point)

    idy_msk = 2*idy_msk
    idx_msk = 2*idx_msk
    if np.random.randint(2) == 1:
        idy_msk += 1
    if np.random.randint(2) == 1:
        idx_msk += 1

    return idy_msk, idx_msk


def generate_mask_n2v(image: torch.Tensor, ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a blind spots mask fot the patch image by randomly switch pixels values

    :param image: Image patch to add blind spots
    :param ratio: Ratio of blind spots for input patch masking
    :return: the transformed image and the mask image
    """
    img_shape = image.shape
    size_window = (5, 5)
    num_sample = int(img_shape[-2] * img_shape[-1] * ratio)

    mask = torch.zeros((img_shape[-2], img_shape[-1]), dtype=torch.float32)
    output = image.clone()

    idy_msk, idx_msk = generate_2d_points((img_shape[-2], img_shape[-1]), num_sample)
    num_sample = len(idy_msk)

    idy_neigh = np.random.randint(-size_window[0] // 2 + size_window[0] % 2,
                                  size_window[0] // 2 + size_window[0] % 2,
                                  num_sample)
    idx_neigh = np.random.randint(-size_window[1] // 2 + size_window[1] % 2,
                                  size_window[1] // 2 + size_window[1] % 2,
                                  num_sample)

    idy_msk_neigh = idy_msk + idy_neigh
    idx_msk_neigh = idx_msk + idx_neigh

    idy_msk_neigh = (idy_msk_neigh + (idy_msk_neigh < 0) * size_window[0] -
                     (idy_msk_neigh >= img_shape[-2]) * size_window[0])
    idx_msk_neigh = (idx_msk_neigh + (idx_msk_neigh < 0) * size_window[1] -
                     (idx_msk_neigh >= img_shape[-1]) * size_window[1])

    id_msk = (idy_msk, idx_msk)
    #id_msk_neigh = (idy_msk_neigh, idx_msk_neigh)

    output[:, :, idy_msk, idx_msk] = image[:, :, idy_msk_neigh, idx_msk_neigh]
    mask[id_msk] = 1.0

    return output, mask


class N2VDeconLoss(torch.nn.Module):
    """MSE Loss with mask for Noise2Void deconvolution

    :param psf_file: File image containing the Point Spread Function
    :return: Loss tensor
    """
    def __init__(self,
                 psf_image: torch.Tensor):
        super().__init__()

        self.__psf = psf_image
        if self.__psf.ndim > 2:
            raise ValueError('N2VDeconLoss PSF must be a gray scaled 2D image')

        self.__psf = self.__psf.view((1, 1, *self.__psf.shape))
        print('psf shape=', self.__psf.shape)
        self.__conv_op = torch.nn.Conv2d(1, 1,
                                         kernel_size=self.__psf.shape[2],
                                         stride=1,
                                         padding=int((self.__psf.shape[2] - 1) / 2),
                                         bias=False)
        with torch.no_grad():
            self.__conv_op.weight = torch.nn.Parameter(self.__psf)

    def forward(self, predict: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        """Calculate forward loss

        :param predict: tensor predicted by the model
        :param target: Reference target tensor
        :param mask: Mask to select pixels of interest
        """
        conv_img = self.__conv_op(predict)

        num = torch.sum((conv_img*mask - target*mask)**2)
        den = torch.sum(mask)
        return num/den


class Noise2VoidDeconv(NNModule):
    """Deconvolution using the noise to void algorithm"""
    def fit(self,
            train_directory: Path,
            val_directory: Path,
            n_channel_in: int = 1,
            n_channels_layer: list[int] = (32, 64, 128),
            patch_size: int = 32,
            n_epoch: int = 25,
            learning_rate: float = 1e-3,
            out_dir: Path = None,
            psf: torch.Tensor = None
            ):
        """Train a model on a dataset
        
        :param train_directory: Directory containing the images used for 
                                training. One file per image,
        :param val_directory: Directory containing the images used for validation of 
                              the training. One file per image,
        :param psf: Point spread function for deconvolution, 
        :param n_channel_in: Number of channels in the input images
        :param n_channels_layer: Number of channels for each hidden layers of the model,
        :param patch_size: Size of square patches used for training the model,
        :param n_epoch: Number of epochs,
        :param learning_rate: Adam optimizer learning rate 
        """
        self._init_model(n_channel_in, n_channel_in, n_channels_layer)
        self._out_dir = out_dir
        self._loss_fn = N2VDeconLoss(psf.to(self.device()))
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate)
        train_dataset = SelfSupervisedPatchDataset(train_directory,
                                                   patch_size=patch_size,
                                                   stride=int(patch_size/2),
                                                   transform=FlipAugmentation())
        val_dataset = SelfSupervisedDataset(val_directory, transform=VisionScale())
        self._train_data_loader = DataLoader(train_dataset,
                                            batch_size=300,
                                            shuffle=True,
                                            drop_last=True,
                                            num_workers=0)
        self._val_data_loader = DataLoader(val_dataset,
                                          batch_size=1,
                                          shuffle=False,
                                          drop_last=False,
                                          num_workers=0)

        self._train_loop(n_epoch)

    def _train_step(self):
        """Runs one step of training"""
        size = len(self._train_data_loader.dataset)
        self._model.train()
        step_loss = 0
        count_step = 0
        tic = timer()
        for batch, (x, _) in enumerate(self._train_data_loader):
            count_step += 1

            masked_x, mask = generate_mask_n2v(x, 0.1)
            x, masked_x, mask = (x.to(self.device()),
                                masked_x.to(self.device()),
                                mask.to(self.device()))

            # Compute prediction error
            prediction = self._model(masked_x)
            loss = self._loss_fn(prediction, x, mask)
            step_loss += loss

            # Backpropagation
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            # count time
            toc = timer()
            full_time = toc - tic
            total_batch = int(size / len(x))
            remains = full_time * (total_batch - (batch+1)) / (batch+1)

            self._after_train_batch({'loss': loss,
                                    'id_batch': batch+1,
                                    'total_batch': total_batch,
                                    'remain_time': int(remains+0.5),
                                    'full_time': int(full_time+0.5)
                                    })

        if count_step > 0:
            step_loss /= count_step
        self._current_loss = step_loss
        return {'train_loss': step_loss}
