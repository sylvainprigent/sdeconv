"""Command line interface module for 2D to 5D image deconvolution"""

import argparse
from skimage.io import imread, imsave
import torch
import numpy as np

from sdeconv.api import SDeconvAPI


def add_args_to_parser(parser: argparse.ArgumentParser, api: SDeconvAPI):
    """Add all the parameters available in the API as an argument in the parser

    :param parser: Argument parser instance
    :param api: SDeconv Application Programming Interface instance

    """
    for filter_name in api.filters.get_keys():
        params = api.filters.get_parameters(filter_name)
        for key, value in params.items():
            parser.add_argument(f"--{key}", help=value['help'], default=value['default'])


def main():
    """Command line interface entrypoint function"""
    parser = argparse.ArgumentParser(description='2D to 5D image deconvolution',
                                     conflict_handler='resolve')

    parser.add_argument('-i', '--input', help='Input image file', default='.tif')
    parser.add_argument('-m', '--method', help='Deconvolution method', default='wiener')
    parser.add_argument('-o', '--output', help='Output image file', default='.tif')
    parser.add_argument('-p', '--plane', help='Plane by plane deconvolution', default=False)

    api = SDeconvAPI()
    add_args_to_parser(parser, api)
    args = parser.parse_args()

    args_dict = vars(args)

    image = torch.Tensor(np.float32(imread(args.input)))
    out_image = api.deconvolve(image, args.method, args.plane, **args_dict)
    imsave(args.output, out_image.detach().numpy())
