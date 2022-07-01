"""Command line interface module for Point Spread Function generator"""

import argparse
from skimage.io import imsave

from sdeconv.api import SDeconvAPI


def add_args_to_parser(parser, api):
    """Add an argument to the parser

    Parameters
    ----------
    parser: argparse.ArgumentParser
        Argument parser object
    api: SDeconvAPI
        SDeconv Application Programming Interface object

    """
    for filter_name in api.filters.get_keys():
        params = api.psfs.get_parameters(filter_name)
        for key, value in params.items():
            parser.add_argument(f"--{key}", help=value['help'], default=value['default'])


def main():
    """Command line interface entrypoint function"""
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
