"""
SDeconv API

This example shows how to use the SDeconv application programming interface
"""

from sdeconv import data
from sdeconv.api import SDeconvAPI
import matplotlib.pyplot as plt

# instantiate the API
api = SDeconvAPI()

# load image
image = data.celegans()

# Generate a PSF
psf = api.generate_psf('SPSFGaussian', sigma=[1.5, 1.5], shape=[13, 13])

# deconvolution with API
image_decon = api.deconvolve(image, "SWiener", plane_by_plane=False, psf=psf, beta=0.005, pad=13)

# plot the result
plt.figure()
plt.subplot(131)
plt.title('Original')
plt.imshow(image.detach().cpu().numpy(), cmap='gray')
plt.axis('off')

plt.subplot(132)
plt.title('PSF')
plt.imshow(psf.detach().cpu().numpy(), cmap='gray')
plt.axis('off')

plt.subplot(133)
plt.title('Wiener deconvolution')
plt.imshow(image_decon.detach().cpu().numpy(), cmap='gray')
plt.axis('off')

plt.show()
