"""This module implements interface for deconvolution based on neural network"""
from abc import abstractmethod
from pathlib import Path

import torch
from skimage.io import imsave

from ..core import seconds2str
from ..core import SConsoleLogger

from ._unet_2d import UNet2D
from ._transforms import VisionScale


class NNModule(torch.nn.Module):
    """Deconvolution using the noise to void algorithm"""
    def __init__(self):
        super().__init__()

        self._model_args = None
        self._model = None
        self._loss_fn = None
        self._optimizer = None
        self._save_all = True
        self._device = None
        self._out_dir = None
        self._val_data_loader = None
        self._train_data_loader = None
        self._progress = SConsoleLogger()
        self._current_epoch = None
        self._current_loss = None

    @abstractmethod
    def fit(self,
            train_directory: Path,
            val_directory: Path,
            n_channel_in: int = 1,
            n_channels_layer: list[int] = (32, 64, 128),
            patch_size: int = 32,
            n_epoch: int = 25,
            learning_rate: float = 1e-3,
            out_dir: Path = None
            ):
        """Train a model on a dataset
        
        :param train_directory: Directory containing the images used for 
                                training. One file per image,
        :param val_directory: Directory containing the images used for validation of 
                              the training. One file per image,
        :param n_channel_in: Number of channels in the input images
        :param n_channels_layer: Number of channels for each hidden layers of the model,
        :param patch_size: Size of square patches used for training the model,
        :param n_epoch: Number of epochs,
        :param learning_rate: Adam optimizer learning rate 
        """
        raise NotImplementedError('NNModule is abstract')

    def _train_loop(self, n_epoch: int):
        """Run the main train loop (should be called in fit)
        
        :param n_epoch: Number of epochs to run
        """
        for epoch in range(n_epoch):
            self._current_epoch = epoch
            train_data = self._train_step()
            self._after_train_step(train_data)
        self._after_train()

    @abstractmethod
    def _train_step(self):
        """Runs one step of training"""

    def _after_train_batch(self, data: dict[str, any]):
        """Instructions runs after one batch

        :param data: Dictionary of metadata to log or process
        """
        prefix = f"Epoch = {self._current_epoch+1:d}"
        loss_str = f"{data['loss']:.7f}"
        full_time_str = seconds2str(int(data['full_time']))
        remains_str = seconds2str(int(data['remain_time']))
        suffix = str(data['id_batch']) + '/' + str(data['total_batch']) + \
            '   [' + full_time_str + '<' + remains_str + ', loss=' + \
            loss_str + ']     '
        self._progress.progress(data['id_batch'],
                               data['total_batch'],
                               prefix=prefix,
                               suffix=suffix)

    def _after_train_step(self, data: dict):
        """Instructions runs after one train step.

        This method can be used to log data or print console messages

        :param data: Dictionary of metadata to log or process
        """
        if self._save_all:
            self.save(Path(self._out_dir, f'model_{self._current_epoch}.ml'))

    def _after_train(self):
        """Instructions runs after the train."""
        # create the output dir
        predictions_dir = self._out_dir / 'predictions'
        predictions_dir.mkdir(parents=True, exist_ok=True)

        # predict on all the test set
        self._model.eval()
        for x, names in self._val_data_loader:
            x = x.to(self.device())

            with torch.no_grad():
                prediction = self._model(x)
            for i, name in enumerate(names):
                imsave(predictions_dir / f'{name}.tif',
                       prediction[i, ...].cpu().numpy())

    def _init_model(self,
                     n_channel_in: int = 1,
                     n_channel_out: int = 1,
                     n_channels_layer: list[int] = (32, 64, 128)
                     ):
        """Initialize the model
        
        :param n_channel_in: Number of channels for the input image
        :param n_channel_in: Number of channels for the output image
        :param n_channels_layer: Number of channels for each layers of the UNet
        """
        self._model_args = {
            "n_channel_in": n_channel_in, 
            "n_channel_out": n_channel_out, 
            "n_channels_layer": n_channels_layer
        }
        self._model = UNet2D(n_channel_in, n_channel_out, n_channels_layer, True)
        self._model.to(self.device())

    def device(self) -> str:
        """Get the GPU if exists
        
        :return: The device name (cuda or CPU)
        """
        if self._device is None:
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return self._device

    def load(self, filename: Path):
        """Load pre-trained model from file
        
        :param: Path of the model file
        """
        params = torch.load(filename, map_location=torch.device(self.device()))
        self._init_model(**params["model_args"])
        self._model.load_state_dict(params['model_state_dict'])
        self._model.to(self.device())

    def save(self, filename: Path):
        """Save the model into file
        
        :param: Path of the model file
        """
        torch.save({
            'model_args': self._model_args,
            'model_state_dict': self._model.state_dict(),
        }, filename)


    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Apply the model on a image or batch of image
        
        :param image: Blurry image for a single channel or batch [(B) Y X]
        :return: deblurred image [(B) Y X]
        """
        if image.ndim == 2:
            image = image.view(1, 1, *image.shape)
        elif image.ndim > 2:
            raise ValueError("The current implementation of neural network "
                             "deconvolution works only with 2D images")

        self._model.eval()
        scaler = VisionScale()
        with torch.no_grad():
            return self._model(scaler(image))
