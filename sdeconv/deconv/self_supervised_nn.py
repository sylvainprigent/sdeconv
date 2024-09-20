"""This module implements self supervised deconvolution with Spitfire regularisation"""
from pathlib import Path
from timeit import default_timer as timer

import torch
from torch.utils.data import DataLoader

from .interface_nn import NNModule
from ._datasets import SelfSupervisedPatchDataset
from ._datasets import SelfSupervisedDataset
from .spitfire import hv_loss


class DeconSpitfireLoss(torch.nn.Module):
    """MSE LOSS with a (de)convolution filter and Spitfire regularisation

    :param psf_file: File containing the PSF for deconvolution
    :return: Loss tensor
    """
    def __init__(self,
                 psf: torch.Tensor,
                 regularization: float = 1e-3,
                 weighting: float = 0.6
                 ):
        super().__init__()
        self.__psf = psf
        self.regularization = regularization
        self.weighting = weighting

        if self.__psf.ndim > 2:
            raise ValueError('DeconMSE PSF must be a gray scaled 2D image')

        self.__psf = self.__psf.view((1, 1, *self.__psf.shape))
        print('psf shape=', self.__psf.shape)
        self.__conv_op = torch.nn.Conv2d(1, 1,
                                         kernel_size=self.__psf.shape[2],
                                         stride=1,
                                         padding=int((self.__psf.shape[2] - 1) / 2),
                                         bias=False)
        with torch.no_grad():
            self.__conv_op.weight = torch.nn.Parameter(self.__psf, requires_grad=False)
        self.__conv_op.requires_grad_(False)

    def forward(self, input_image: torch.Tensor, target: torch.Tensor):
        """Deconvolution L2 data-term

        Compute the L2 error between the original image (input) and the
        convoluted reconstructed image (target)

        :param input_image: Tensor of shape BCYX containing the original blurry image
        :param target: Tensor of shape BCYX containing the estimated deblurred image
        """
        conv_img = self.__conv_op(input_image)
        mse = torch.nn.MSELoss()
        return self.regularization*mse(target, conv_img) + \
                   (1-self.regularization)*hv_loss(input_image, weighting=self.weighting)


class SelfSupervisedNNDeconv(NNModule):
    """Deconvolution using a neural network trained using the Spitfire loss"""
    def fit(self,
            train_directory: Path,
            val_directory: Path,
            n_channel_in: int = 1,
            n_channels_layer: list[int] = (32, 64, 128),
            patch_size: int = 32,
            n_epoch: int = 25,
            learning_rate: float = 1e-3,
            out_dir: Path = None,
            weight: float = 0.9,
            reg: float = 0.95,
            psf: torch.Tensor = None
            ):
        """Train a model on a dataset
        
        :param train_directory: Directory containing the images used 
                                for training. One file per image,
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
        self._loss_fn = DeconSpitfireLoss(psf.to(self.device()), reg, weight)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate)
        train_dataset = SelfSupervisedPatchDataset(train_directory,
                                                   patch_size=patch_size,
                                                   stride=int(patch_size/2))
        val_dataset = SelfSupervisedDataset(val_directory)
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

            x = x.to(self.device())

            # Compute prediction error
            prediction = self._model(x)
            loss = self._loss_fn(prediction, x)
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
