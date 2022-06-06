import os
import argparse

from sdeconv.api import SDeconvAPI
from skimage.io import imread, imsave


def add_args_to_parser(parser, api):
    for filter_name in api.filters.get_keys():
        params = api.psfs.get_parameters(filter_name)
        for key, value in params.items():
            parser.add_argument(f"--{key}", help=value['help'], default=value['default'])


def main():
    parser = argparse.ArgumentParser(description='2D to 5D image deconvolution',
                                     conflict_handler='resolve')

    parser.add_argument('-m', '--method', help='Deconvolution method', default='wiener')
    parser.add_argument('-o', '--output', help='Output image file', default='.tif')

    api = SDeconvAPI()
    add_args_to_parser(parser, api)
    args = parser.parse_args()

    args_dict = vars(args)
    out_image = api.psf(args.method, **args_dict)
    imsave(args.output, out_image.detach().numpy())
