"""Module to implement a UNet in 2D wwith pytorch module"""
import torch
from torch import nn


class UNetConvBlock(nn.Module):
    """Convolution block for UNet architecture

    This block is 2 convolution layers with a ReLU.
    An optional batch norm can be added after each convolution layer

    :param n_channels_in: Number of input channels (or features)
    :param n_channels_out: Number of output channels (or features)
    :param use_batch_norm: True to use the batch norm layers
    """
    def __init__(self,
                 n_channels_in: int,
                 n_channels_out: int,
                 use_batch_norm: bool = True):
        super().__init__()

        self.use_batch_norm = use_batch_norm
        self.conv1 = nn.Conv2d(n_channels_in, n_channels_out,
                               kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(n_channels_out)

        self.conv2 = nn.Conv2d(n_channels_out, n_channels_out,
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(n_channels_out)

        self.relu = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the model

        :param inputs: Data to process
        :return: The processed data
        """
        x = self.conv1(inputs)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = self.relu(x)

        return x


class UNetEncoderBlock(nn.Module):
    """Encoder block of the UNet architecture

    The encoder block is a convolution block and a max polling layer

    :param n_channels_in: Number of input channels (or features)
    :param n_channels_out: Number of output channels (or features)
    :param use_batch_norm: True to use the batch norm layers
    """
    def __init__(self,
                 n_channels_in: int,
                 n_channels_out: int,
                 use_batch_norm: bool = True):
        super().__init__()

        self.conv = UNetConvBlock(n_channels_in, n_channels_out,
                                  use_batch_norm)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs: torch.Tensor):
        """torch module forward method

        :param inputs: tensor to process
        """
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p


class UNetDecoderBlock(nn.Module):
    """Decoder block of a UNet architecture

    The decoder is an up-sampling concatenation and convolution block

    :param n_channels_in: Number of input channels (or features)
    :param n_channels_out: Number of output channels (or features)
    :param use_batch_norm: True to use the batch norm layers
    """
    def __init__(self,
                 n_channels_in: int,
                 n_channels_out: int,
                 use_batch_norm: bool = True):
        super().__init__()

        self.up = nn.Upsample(scale_factor=(2, 2), mode='nearest')
        self.conv = UNetConvBlock(n_channels_in+n_channels_out,
                                  n_channels_out, use_batch_norm)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor):
        """Module torch forward

        :param inputs: input tensor
        :param skip: skip connection tensor
        """
        x = self.up(inputs)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)

        return x


class UNet2D(nn.Module):
    """Implementation of a UNet network

    :param n_channels_in: Number of input channels (or features),
    :param n_channels_out: Number of output channels (or features),
    :param n_feature_first: Number of channels (or features) in the first convolution block,
    :param use_batch_norm: True to use the batch norm layers
    """
    def __init__(self,
                 n_channels_in: int = 1,
                 n_channels_out: int = 1,
                 n_channels_layers: list[int] = (32, 64, 128),
                 use_batch_norm: bool = False):
        super().__init__()

        # Encoder
        self.encoder = nn.ModuleList()
        for idx, n_channels in enumerate(n_channels_layers[:-1]):
            n_in = n_channels_in if idx == 0 else n_channels_layers[idx-1]
            n_out = n_channels
            self.encoder.append(UNetEncoderBlock(n_in, n_out, use_batch_norm))

        # Bottleneck
        self.bottleneck = UNetConvBlock(n_channels_layers[-2],
                                        n_channels_layers[-1],
                                        use_batch_norm)

        # Decoder
        self.decoder = nn.ModuleList()
        for idx in reversed(range(len(n_channels_layers))):
            if idx > 0:
                n_in = n_channels_layers[idx]
                n_out = n_channels_layers[idx-1]
                self.decoder.append(UNetDecoderBlock(n_in, n_out, use_batch_norm))

        self.outputs = nn.Conv2d(n_channels_layers[0], n_channels_out,
                                 kernel_size=1, padding=0)

        self.num_layers = len(n_channels_layers)-1


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Module torch forward

        :param inputs: input tensor
        :return: the tensor processed by the network
        """
        # Encoder
        skips = []
        p = [None] * (len(self.encoder)+1)
        p[0] = inputs
        for idx, layer in enumerate(self.encoder):
            s, p[idx+1] = layer(p[idx])
            skips.append(s)

        # Bottleneck
        d = [None] * (len(self.decoder)+1)
        d[0] = self.bottleneck(p[-1])

        # decoder
        for idx, layer in enumerate(self.decoder):
            d[idx+1] = layer(d[idx], skips[self.num_layers-idx-1])

        # Classifier
        return self.outputs(d[-1])
