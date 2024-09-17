"""Implementation of deconvolution using noise2void algorithm"""
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
import torch
from torch.utils.data import DataLoader

from skimage.io import imsave

from ..core import seconds2str
from ..core import SConsoleLogger
from ._unet_2d import UNet2D
from ._datasets import SelfSupervisedPatchDataset
from ._datasets import SelfSupervisedDataset

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
    id_msk_neigh = (idy_msk_neigh, idx_msk_neigh)

    output[:, :, *id_msk] = image[:, :, *id_msk_neigh]
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
    

class Noise2VoidDeconv(torch.nn.Module):
    """Deconvolution using the noise to void algorithm"""
    def __init__(self):
        super().__init__()

        self.__model_args = None
        self.__model = None
        self.loss_fn = None
        self.optimizer = None
        self.save_all = True
        self.__device = None
        self.out_dir = None
        self.progress = SConsoleLogger()

    def fit(self, 
            train_directory: Path,
            val_directory: Path,
            psf: torch.Tensor, 
            n_channel_in: int = 1,
            n_channels_layer: list[int] = [32, 64, 128],
            patch_size: int = 32,
            n_epoch: int = 25,
            learning_rate: float = 1e-3, 
            out_dir: Path = None
            ):
        """Train a model on a dataset
        
        :param train_directory: Directory containing the images used for training. One file per image,
        :param val_directory: Directory containing the images used for validation of the training. One file per image,
        :param psf: Point spread function for deconvolution, 
        :param n_channel_in: Number of channels in the input images
        :param n_channels_layer: Number of channels for each hidden layers of the model,
        :param patch_size: Size of square patches used for training the model,
        :param n_epoch: Number of epochs,
        :param learning_rate: Adam optimizer learning rate 
        """
        self.__init_model(n_channel_in, n_channel_in, n_channels_layer)
        self.out_dir = out_dir
        self.loss_fn = N2VDeconLoss(psf.to(self.device()))
        self.optimizer = torch.optim.Adam(self.__model.parameters(), lr=learning_rate)
        train_dataset = SelfSupervisedPatchDataset(train_directory, patch_size=patch_size, stride=int(patch_size/2))
        val_dataset = SelfSupervisedDataset(val_directory)
        self.train_data_loader = DataLoader(train_dataset,
                                            batch_size=300,
                                            shuffle=True,
                                            drop_last=True,
                                            num_workers=0)
        self.val_data_loader = DataLoader(val_dataset,
                                          batch_size=1,
                                          shuffle=False,
                                          drop_last=False,
                                          num_workers=0)
        
        for epoch in range(n_epoch):
            self.current_epoch = epoch
            train_data = self.__train_step()
            self.__after_train_step(train_data)
        self.__after_train()

    def __train_step(self):
        """Runs one step of training"""
        size = len(self.train_data_loader.dataset)
        self.__model.train()
        step_loss = 0
        count_step = 0
        tic = timer()
        for batch, (x, _) in enumerate(self.train_data_loader):
            count_step += 1

            masked_x, mask = generate_mask_n2v(x, 0.1)
            x, masked_x, mask = x.to(self.device()), masked_x.to(self.device()), mask.to(self.device())

            # Compute prediction error
            prediction = self.__model(masked_x)
            loss = self.loss_fn(prediction, x, mask)
            step_loss += loss

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # count time
            toc = timer()
            full_time = toc - tic
            total_batch = int(size / len(x))
            remains = full_time * (total_batch - (batch+1)) / (batch+1)

            self.__after_train_batch({'loss': loss,
                                    'id_batch': batch+1,
                                    'total_batch': total_batch,
                                    'remain_time': int(remains+0.5),
                                    'full_time': int(full_time+0.5)
                                    })

        if count_step > 0:
            step_loss /= count_step
        self.current_loss = step_loss
        return {'train_loss': step_loss}
    
    def __after_train_batch(self, data: dict[str, any]):
        """Instructions runs after one batch

        :param data: Dictionary of metadata to log or process
        """
        prefix = f"Epoch = {self.current_epoch+1:d}"
        loss_str = f"{data['loss']:.7f}"
        full_time_str = seconds2str(int(data['full_time']))
        remains_str = seconds2str(int(data['remain_time']))
        suffix = str(data['id_batch']) + '/' + str(data['total_batch']) + \
            '   [' + full_time_str + '<' + remains_str + ', loss=' + \
            loss_str + ']     '
        self.progress.progress(data['id_batch'],
                               data['total_batch'],
                               prefix=prefix,
                               suffix=suffix)
        
    def __after_train_step(self, data: dict):
        """Instructions runs after one train step.

        This method can be used to log data or print console messages

        :param data: Dictionary of metadata to log or process
        """
        if self.save_all:
            self.save(Path(self.out_dir, f'model_{self.current_epoch}.ml'))
            
    def __after_train(self):
        """Instructions runs after the train."""
        # create the output dir
        predictions_dir = self.out_dir / 'predictions'
        predictions_dir.mkdir(parents=True, exist_ok=True)

        # predict on all the test set
        self.__model.eval()
        for x, names in self.val_data_loader:
            x = x.to(self.device())

            with torch.no_grad():
                prediction = self.__model(x)
            for i, name in enumerate(names):
                imsave(predictions_dir / f'{name}.tif',
                       prediction[i, ...].cpu().numpy())

    def __init_model(self, 
                     n_channel_in: int = 1, 
                     n_channel_out: int = 1, 
                     n_channels_layer: list[int] = [32, 64, 128]
                     ):
        """Initialize the model
        
        :param n_channel_in: Number of channels for the input image
        :param n_channel_in: Number of channels for the output image
        :param n_channels_layer: Number of channels for each layers of the UNet
        """
        self.__model_args = {
            "n_channel_in": n_channel_in, 
            "n_channel_out": n_channel_out, 
            "n_channels_layer": n_channels_layer
        }
        self.__model = UNet2D(n_channel_in, n_channel_out, n_channels_layer, True)

    def device(self) -> str:
        """Get the GPU if exists
        
        :return: The device name (cuda or CPU)
        """
        if self.__device is None:
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return self.__device

    def load(self, filename: Path):
        """Load pre-trained model from file
        
        :param: Path of the model file
        """
        params = torch.load(filename, map_location=torch.device(self.device()))
        self.__init_model(**params["model_args"])
        self.__model.load_state_dict(params['model_state_dict'])
        self.__model.to(self.device())

    def save(self, filename: Path):
        """Save the model into file
        
        :param: Path of the model file
        """
        torch.save({
            'model_args': self.__model_args,
            'model_state_dict': self.__model.state_dict(),
        }, filename)


    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Apply the model on a image or batch of image
        
        :param image: Blurry image for a single channel or batch [(B) Y X]
        :return: deblurred image [(B) Y X]
        """
        if image.ndim == 2:
            image = image.view(1, *image.shape)
        if image.ndim > 2:
            raise ValueError("The current implementation of Noise2VoidDeconv takes only on 2D images")

        with torch.no_grad():
            return self.__model()
