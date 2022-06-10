"""
Richardson-Lucy deconvolution

This example shows how to use the Richardson-Lucy deconvolution on a 2D image
"""

import matplotlib.pyplot as plt
from sdeconv.data import celegans
from sdeconv.psfs import SPSFGaussian
from sdeconv.deconv import SRichardsonLucy


# load a 2D sample
image = celegans()

# Generate a 2D PSF
psf_generator = SPSFGaussian((1.5, 1.5), (13, 13))
psf = psf_generator()

# apply Wiener filter
filter_ = SRichardsonLucy(psf, niter=30, pad=13)
out_image = filter_(image)

# display results
plt.figure()
plt.title('PSF')
plt.imshow(psf.detach().numpy(), cmap='gray')

plt.figure()
plt.title('C. elegans original')
plt.imshow(image.detach().numpy(), cmap='gray')

plt.figure()
plt.title('C. elegans Richardson-Lucy')
plt.imshow(out_image.detach().numpy(), cmap='gray')

plt.show()

