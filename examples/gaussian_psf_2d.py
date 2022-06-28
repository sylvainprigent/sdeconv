"""
Gaussian PSF

This example shows how to generate a Gaussian PSF
"""

import matplotlib.pyplot as plt
from sdeconv.psfs import SPSFGaussian


psf_generator = SPSFGaussian(sigma=(1.5, 1.5), shape=(13, 13))
psf = psf_generator()

plt.figure()
plt.title('Gaussian PSF')
plt.imshow(psf.detach().numpy(), cmap='gray')
plt.show()