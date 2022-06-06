import os
import argparse

from sdeconv.api import SDeconvAPI
from skimage.io import imread, imsave


def add_args_to_parser(parser, api):
    for filter_name in api.filters.get_keys():
        params = api.filters.get_parameters(filter_name)
        for key, value in params.items():
            parser.add_argument(f"--{key}", help=value['help'], default=value['default'])


def main():
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
