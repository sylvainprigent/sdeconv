Quick start
===========

This is a quick start example of how to use the **SDeconv** library. This section supposes you to know the principles
of deconvolution. If it is not the case, please refer to the
`Guide <guide>`_.

Input images
------------
Input images are 2D or 3D gray scaled images. 2D images are represented as numpy arrays with the following
columns ordering ``[Y, X]`` and 3D images are represented with numpy array with ``[Z, Y, X]`` columns ordering

First we load the images:

.. code-block:: python3

    from sdeconv import data

    image = data.celegans()


Deconvolution using the API
---------------------------

.. code-block:: python3

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


Deconvolution using the library classes
---------------------------------------

.. code-block:: python3

    import matplotlib.pyplot as plt
    from sdeconv.data import celegans
    from sdeconv.psfs import SPSFGaussian
    from sdeconv.deconv import SWiener

    # load a 2D sample
    image = celegans()

    # Generate a 2D PSF
    psf_generator = SPSFGaussian((1.5, 1.5), (13, 13))
    psf = psf_generator()

    # apply Wiener filter
    wiener = SWiener(psf, beta=0.005, pad=13)
    out_image = wiener(image)

    # display results
    plt.figure()
    plt.title('PSF')
    plt.imshow(psf.detach().numpy(), cmap='gray')

    plt.figure()
    plt.title('C. elegans original')
    plt.imshow(image.detach().numpy(), cmap='gray')

    plt.figure()
    plt.title('C. elegans Wiener')
    plt.imshow(out_image.detach().numpy(), cmap='gray')

    plt.show()