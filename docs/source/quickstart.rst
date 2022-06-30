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

.. code-clock:: python3

    from sdeconv import data
    from sdeconv.api import SDeconvAPI
    import matplotlib.pyplot as plt

    # instantiate the API
    api = SDeconvAPI()

    # load image
    image = data.celegans()

    # Generate a PSF
    psf = api.psf('SPSFGaussian', sigma=1.5)

    # deconvolution with API
    image_decon = api.deconvolve(image, "wiener", plane_by_plane=False, psf=psf, beta=0.005, pad=13)

    # plot the result
    plt.figure()
    plt.subplot(131)
    plt.title('Original')
    plt.imshow(image.detach().cpu().numpy()[0, ...], cmap='gray')
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

Particles detections
--------------------
The first step of particle tracking is to detect individual particles frame by frame.
**STracking** provides ``SDetector`` interface for particles detector. In this example we detect particles with the
*Difference of Gaussians* detector:

.. code-block:: python3

    from stracking.detectors import DoGDetector

    detector = DoGDetector(min_sigma=4, max_sigma=5, threshold=0.2)
    particles = detector.run(image)


The output ```articles`` is an instance of the ``SParticles`` container. It contains the list of particles as a numpy
array, the properties of the particles as a *dict* and the image scale as a *tuple*

Particles linking
-----------------
The second step is linking the particles to create tracks.
**STracking** provides ``SLinker`` interface to implement mulitple linking algorithms. In this quick start, we use the
*Shorted path* graph based linker, using the Euclidean distance between particles as a link cost function:

.. code-block:: python3

    from stracking.linkers import SPLinker, EuclideanCost

    euclidean_cost = EuclideanCost(max_cost=3000)
    my_tracker = SPLinker(cost=euclidean_cost, gap=1)
    tracks = my_tracker.run(particles)


The output ``tracks`` in an instance of the ``STracks`` container. It contains the list of tracks as a numpy array and
all the tracks metadata in dictionaries.

The next steps show the usage of ``SProperty``, ``SFeature`` and ``SFilter`` to analyse the trajectories

Particles properties
--------------------
The tracks properties module allows to calculate properties of the particles. This quickstart example
shows how to calculate the intensity properties of particles:

.. code-block:: python3

    from stracking.properties import IntensityProperty

    property_calc = IntensityProperty(radius=2)
    property_calc.run(particles, image)

All the calculated properties are saved in the properties attribute of the ``SParticles`` container.

Tracks features
---------------
The tracks features module allows to calculate features of tracks like length and distance. This quickstart example shows how
to calculate the distance of tracks:

.. code-block:: python3

    from stracking.features import DistanceFeature

    feature_calc = DistanceFeature()
    feature_calc.run(tracks)

The calculated features are stored in the ``features`` attribute of the ``STracks`` container.

Tracks filter
-------------
The last part is the filter module. It allows to extract a subset of tracks base on a defined criterion. In this example, we select the tracks that move less that a distance of 60 pixels:

.. code-block:: python3

    from stracking.filters import FeatureFilter

    filter = FeatureFilter(feature_name='distance', min_val=0, max_val=60)
    filtered_tracks = filter.run(tracks)

Filtered set of tracks are return as a ``STracks`` object.
