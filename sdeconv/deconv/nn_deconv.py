"""This module implements self supervised deconvolution with Spitfire regularisation"""
from pathlib import Path
from timeit import default_timer as timer

import torch
from torch.utils.data import DataLoader
from skimage.io import imsave

from .interface_nn import NNModule
from ._datasets import RestorationDataset
from ._datasets import RestorationPatchDatasetLoad
from ._datasets import RestorationPatchDataset
from ._transforms import FlipAugmentation, VisionScale


class NNDeconv(NNModule):
    """Deconvolution using a neural network trained using ground truth"""
    def fit(self,
            train_directory: Path,
            val_directory: Path,
            n_channel_in: int = 1,
            n_channels_layer: list[int] = (32, 64, 128),
            patch_size: int = 32,
            n_epoch: int = 25,
            learning_rate: float = 1e-3,
            out_dir: Path = None,
            preload: bool = True
            ):
        """Train a model on a dataset
        
        :param train_directory: Directory containing the images used for 
                                training. One file per image,
        :param val_directory: Directory containing the images used for validation of the 
                              training. One file per image,
        :param n_channel_in: Number of channels in the input images
        :param n_channels_layer: Number of channels for each hidden layers of the model,
        :param patch_size: Size of square patches used for training the model,
        :param n_epoch: Number of epochs,
        :param learning_rate: Adam optimizer learning rate 
        """
        self._init_model(n_channel_in, n_channel_in, n_channels_layer)
        self._out_dir = out_dir
        self._loss_fn = torch.nn.MSELoss()
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate)
        if preload:
            train_dataset = RestorationPatchDatasetLoad(train_directory / "source",
                                                    train_directory / "target",
                                                    patch_size=patch_size,
                                                    stride=int(patch_size/2),
                                                    transform=FlipAugmentation())
        else:
            train_dataset = RestorationPatchDataset(train_directory / "source",
                                                    train_directory / "target",
                                                    patch_size=patch_size,
                                                    stride=int(patch_size/2),
                                                    transform=FlipAugmentation())
        val_dataset = RestorationDataset(val_directory / "source",
                                         val_directory / "target",
                                         transform=VisionScale())
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
        for batch, (x, y, _) in enumerate(self._train_data_loader):
            count_step += 1

            x = x.to(self.device())

            # Compute prediction error
            prediction = self._model(x)
            loss = self._loss_fn(prediction, y)
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

    def _after_train(self):
        """Instructions runs after the train."""
        # create the output dir
        predictions_dir = self._out_dir / 'predictions'
        predictions_dir.mkdir(parents=True, exist_ok=True)

        # predict on all the test set
        self._model.eval()
        for x, _, names in self._val_data_loader:
            x = x.to(self.device())

            with torch.no_grad():
                prediction = self._model(x)
            for i, name in enumerate(names):
                imsave(predictions_dir / f'{name}.tif',
                       prediction[i, ...].cpu().numpy())
