"""
Gaussian PSF

This example shows how to generate a Gaussian PSF
"""

from sdeconv.psfs import SPSFGaussian
import napari
import time

psf_generator = SPSFGaussian(sigma=(0.5, 1.5, 1.5), shape=(25, 128, 128))
t = time.time()
psf = psf_generator()
elapsed = time.time() - t
print('elapsed = ', elapsed)

viewer = napari.view_image(psf.detach().numpy(), scale=[200, 100, 100])
napari.run()
