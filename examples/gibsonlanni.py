"""
Gibson Lanni PSF

This example shows how to generate a Gibson Lanni PSF
"""

import matplotlib.pyplot as plt
from sdeconv.psfs import SPSFGibsonLanni
import napari
import time

psf_generator = SPSFGibsonLanni((11, 128, 128), use_square=True)
t = time.time()

psf = psf_generator()
elapsed = time.time() - t
print('elapsed = ', elapsed)

viewer = napari.view_image(psf.detach().numpy(), scale=[200, 100, 100])
napari.run()
