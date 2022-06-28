"""
Spitfire deconvolution

This example shows how to use the Spitfire deconvolution on a 2D image
"""

import matplotlib.pyplot as plt
from sdeconv.data import celegans
from sdeconv.psfs import SPSFGaussian
from sdeconv.deconv import Spitfire


# load a 2D sample
image = celegans()

# Generate a 2D PSF
psf_generator = SPSFGaussian((1.5, 1.5), (13, 13))
psf = psf_generator()

# apply Spitfire filter
filter_ = Spitfire(psf, weight=0.6, reg=0.995, gradient_step=0.01, precision=1e-7, pad=13)
out_image = filter_(image)

# display results
plt.figure()
plt.title('PSF')
plt.imshow(psf.detach().numpy(), cmap='gray')

plt.figure()
plt.title('C. elegans original')
plt.imshow(image.detach().numpy(), cmap='gray')

plt.figure()
plt.title('C. elegans Spitfire')
plt.imshow(out_image.detach().numpy(), cmap='gray')

plt.show()

