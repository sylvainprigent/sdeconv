Quick start
===========

This is a quick start example of how to use the **SDeconv** library. This section supposes you to know the principles
of deconvolution. If it is not the case, please refer to the
:doc:`References <references>`.

Input images
------------
Input images are 2D or 3D gray scaled images. 2D images are represented as torch tensors with the following
columns ordering ``[Y, X]`` and 3D images are represented with torch tensors with ``[Z, Y, X]`` columns ordering

Sample images can be loaded using the data module:

.. code-block:: python3

    from sdeconv import data

    image = data.celegans()


Deconvolution using the API
---------------------------

Bellow is an example how to write a deconvolution script with the API. In this example, we run the Wiener deconvolution algorithm:

.. code-block:: python3

    from sdeconv import data
    from sdeconv.api import SDeconvAPI
    import matplotlib.pyplot as plt

    # Instantiate the API
    api = SDeconvAPI()

    # Load image
    image = data.celegans()

    # Generate a PSF
    psf = api.generate_psf('SPSFGaussian', sigma=[1.5, 1.5], shape=[13, 13])

    # Deconvolution with API
    image_decon = api.deconvolve(image, "SWiener", plane_by_plane=False, psf=psf, beta=0.005, pad=13)

    # Plot the results
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


The advantage of using the API is that it implements several strategies to deconvolve a 3D or 3D+t image either plane 
by plane then frame by frame, or frame by frame in 3D.


Deconvolution using the library classes
---------------------------------------

When we need only one method, the easiest way may be to call direclty the class that implements the deconvolution 
algorithm:

.. code-block:: python3

    import matplotlib.pyplot as plt
    from sdeconv.data import celegans
    from sdeconv.psfs import SPSFGaussian
    from sdeconv.deconv import SWiener

    # Load a 2D sample
    image = celegans()

    # Generate a 2D PSF
    psf_generator = SPSFGaussian((1.5, 1.5), (13, 13))
    psf = psf_generator()

    # Apply Wiener filter
    wiener = SWiener(psf, beta=0.005, pad=13)
    out_image = wiener(image)

    # Display results
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

Please refer to :doc:`Modules <modules>` for more details on the interfaces and the list of available PSFs and deconvolution methods. 
