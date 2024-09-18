"""This module implements the datasets for deep learning training"""
from typing import Callable
from pathlib import Path

import numpy as np
import torch
from skimage.io import imread

from torch.utils.data import Dataset


class SelfSupervisedPatchDataset(Dataset):
    """Gray scaled image patched dataset for Self supervised learning

    :param images_dir: Directory containing the training images
    :param patch_size: Size of the squared training patches
    :param stride: Stride used to extract overlapping patches from images
    :param transform: Transformation to images before model
    """
    def __init__(self,
                 images_dir: Path,
                 patch_size: int = 40,
                 stride: int = 10,
                 transform: Callable = None):
        super().__init__()
        self.images_dir = Path(images_dir)
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform

        source_images = sorted(self.images_dir.glob('*.*'))

        self.nb_images = len(source_images)
        image = imread(source_images[0])
        self.n_patches = self.nb_images * ((image.shape[0] - patch_size) // stride) * \
                                          ((image.shape[1] - patch_size) // stride)
        print('num patches = ', self.n_patches)

        # Load all the images in a list
        self.images_data = []
        for source in source_images:
            self.images_data.append(np.float32(imread(source)))

    def __len__(self):
        return self.n_patches

    def __getitem__(self, idx):
        nb_patch_per_img = self.n_patches // self.nb_images

        img_number = idx // nb_patch_per_img

        img_np = self.images_data[img_number]

        nb_patch_w = (img_np.shape[1] - self.patch_size) // self.stride
        idx = idx % nb_patch_per_img
        i, j = idx // nb_patch_w, idx % nb_patch_w
        img_patch = \
            img_np[i * self.stride:i * self.stride + self.patch_size,
                   j * self.stride:j * self.stride + self.patch_size]

        if self.transform:
            img_patch = self.transform(torch.Tensor(img_patch))
        else:
            img_patch = torch.Tensor(img_patch).float()

        return (
            img_patch.view(1, *img_patch.shape),
            str(idx)
        )


class SelfSupervisedDataset(Dataset):
    """Gray scaled image dataset for Self supervised learning

    :param images_dir: Directory containing the training images
    :param transform: Transformation to images before model
    """
    def __init__(self,
                 images_dir: Path,
                 transform: Callable = None):
        super().__init__()
        self.images_dir = Path(images_dir)
        self.transform = transform

        self.source_images = sorted(self.images_dir.glob('*.*'))

        self.nb_images = len(self.source_images)

        # Load all the images in a list
        self.images_data = []
        for source in self.source_images:
            self.images_data.append(np.float32(imread(source)))

    def __len__(self):
        return self.nb_images

    def __getitem__(self, idx):

        img_patch = self.images_data[idx]
        if self.transform:
            img_patch = self.transform(torch.Tensor(img_patch))
        else:
            img_patch = torch.Tensor(img_patch).float()

        return (
            img_patch.view(1, *img_patch.shape),
            self.source_images[idx].stem
        )


class RestorationDataset(Dataset):
    """Dataset for image restoration from full images

    All the training images must be saved as individual images in source and
    target folders.

    :param source_dir: Path of the noisy training images (or patches)
    :param target_dir: Path of the ground truth images (or patches)
    :param transform: Transformation to apply to the image before model call
    """
    def __init__(self,
                 source_dir: str | Path,
                 target_dir: str | Path,
                 transform: Callable = None):
        super().__init__()
        self.device = None
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.transform = transform

        self.source_images = sorted(self.source_dir.glob('*.*'))
        self.target_images = sorted(self.target_dir.glob('*.*'))
        if len(self.source_images) != len(self.target_images):
            raise ValueError("Source and target dirs are not the same length")

        self.nb_images = len(self.source_images)

    def __len__(self):
        return self.nb_images

    def __getitem__(self, idx):
        source_patch = np.float32(imread(self.source_images[idx]))
        target_patch = np.float32(imread(self.target_images[idx]))

        # numpy to tensor
        source_patch = torch.from_numpy(source_patch).view(1, *source_patch.shape).float()
        target_patch = torch.from_numpy(target_patch).view(1, *target_patch.shape).float()

        # data augmentation
        if self.transform:
            both_images = torch.cat((source_patch.unsqueeze(0), target_patch.unsqueeze(0)), 0)
            transformed_images = self.transform(both_images)
            source_patch = transformed_images[0, ...]
            target_patch = transformed_images[1, ...]

        return source_patch, target_patch, self.source_images[idx].stem


class RestorationPatchDataset(Dataset):
    """Dataset for image restoration using patches

    All the training images must be saved as individual images in source and
    target folders.

    :param source_dir: Path of the noisy training images (or patches)
    :param target_dir: Path of the ground truth images (or patches)
    :param patch_size: Size of the patches (width=height)
    :param stride: Length of the patch overlapping
    :param transform: Transformation to apply to the image before model call

    """
    def __init__(self,
                 source_dir: str | Path,
                 target_dir: str | Path,
                 patch_size: int = 40,
                 stride: int = 10,
                 transform: Callable = None):
        super().__init__()
        self.device = None
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform

        self.source_images = sorted(self.source_dir.glob('*.*'))
        self.target_images = sorted(self.target_dir.glob('*.*'))
        if len(self.source_images) != len(self.target_images):
            raise ValueError("Source and target dirs are not the same length")

        self.nb_images = len(self.source_images)
        image = imread(self.source_images[0])
        self.n_patches = self.nb_images * ((image.shape[0] - patch_size) // stride) * \
                                          ((image.shape[1] - patch_size) // stride)

    def __len__(self):
        return self.n_patches

    def __getitem__(self, idx):
        # Crop a patch from original image
        nb_patch_per_img = self.n_patches // self.nb_images

        elt = self.source_images[idx // nb_patch_per_img]

        img_source_np = \
            np.float32(imread(self.source_dir / elt))
        img_target_np = \
            np.float32(imread(self.target_dir / elt))

        nb_patch_w = (img_source_np.shape[1] - self.patch_size) // self.stride
        idx = idx % nb_patch_per_img
        i, j = idx // nb_patch_w, idx % nb_patch_w
        source_patch = \
            img_source_np[i * self.stride:i * self.stride + self.patch_size,
            j * self.stride:j * self.stride + self.patch_size]
        target_patch = \
            img_target_np[i * self.stride:i * self.stride + self.patch_size,
            j * self.stride:j * self.stride + self.patch_size]

        # numpy to tensor
        source_patch = torch.from_numpy(source_patch).view(1, *source_patch.shape).float()
        target_patch = torch.from_numpy(target_patch).view(1, *target_patch.shape).float()

        # data augmentation
        if self.transform:
            both_images = torch.cat((source_patch.unsqueeze(0), target_patch.unsqueeze(0)), 0)
            transformed_images = self.transform(both_images)
            source_patch = transformed_images[0, ...]
            target_patch = transformed_images[1, ...]

        # to tensor
        return (source_patch,
                target_patch,
                str(idx)
                )


class RestorationPatchDatasetLoad(Dataset):
    """Dataset for image restoration using patches preloaded in memory

    All the training images must be saved as individual images in source and
    target folders.
    This version load all the dataset in the CPU

    :param source_dir: Path of the noisy training images (or patches)
    :param target_dir: Path of the ground truth images (or patches)
    :param patch_size: Size of the patches (width=height)
    :param stride: Length of the patch overlapping
    :param transform: Transformation to apply to the image before model call

    """
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
    def __init__(self,
                 source_dir: str | Path,
                 target_dir: str | Path,
                 patch_size: int = 40,
                 stride: int = 10,
                 transform: Callable = None):
        super().__init__()
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform

        self.source_images = sorted(self.source_dir.glob('*.*'))
        self.target_images = sorted(self.target_dir.glob('*.*'))

        if len(self.source_images) != len(self.target_images):
            raise ValueError("Source and target dirs are not the same length")

        self.nb_images = len(self.source_images)
        image = imread(self.source_images[0])
        self.n_patches = self.nb_images * ((image.shape[0] - patch_size) // stride) * \
                                          ((image.shape[1] - patch_size) // stride)
        print('num patches = ', self.n_patches)

        # Load all the images in a list
        self.source_data = []
        for source in self.source_images:
            self.source_data.append(np.float32(imread(source)))
        self.target_data = []
        for target in self.target_images:
            self.target_data.append(np.float32(imread(target)))

    def __len__(self):
        return self.n_patches

    def __getitem__(self, idx):
        # Crop a patch from original image
        nb_patch_per_img = self.n_patches // self.nb_images

        img_number = idx // nb_patch_per_img

        img_source_np = self.source_data[img_number]
        img_target_np = self.target_data[img_number]

        nb_patch_w = (img_source_np.shape[1] - self.patch_size) // self.stride
        idx = idx % nb_patch_per_img
        i, j = idx // nb_patch_w, idx % nb_patch_w
        source_patch = \
            img_source_np[i * self.stride:i * self.stride + self.patch_size,
            j * self.stride:j * self.stride + self.patch_size]
        target_patch = \
            img_target_np[i * self.stride:i * self.stride + self.patch_size,
            j * self.stride:j * self.stride + self.patch_size]

        # numpy to tensor
        source_patch = torch.from_numpy(source_patch).view(1, *source_patch.shape).float()
        target_patch = torch.from_numpy(target_patch).view(1, *target_patch.shape).float()

        # data augmentation
        if self.transform:
            both_images = torch.cat((source_patch.unsqueeze(0), target_patch.unsqueeze(0)), 0)
            transformed_images = self.transform(both_images)
            source_patch = transformed_images[0, ...]
            target_patch = transformed_images[1, ...]

        return (source_patch,
                target_patch,
                str(idx)
                )
