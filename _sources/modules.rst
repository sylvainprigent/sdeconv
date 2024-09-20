Modules
=======

Point Spread Functions
----------------------

.. currentmodule:: sdeconv.psfs

.. autosummary::
    :toctree: generated
    :nosignatures:

    SPSFGaussian
    SPSFGibsonLanni
    SPSFLorentz


Deconvolution algorithms
------------------------

.. currentmodule:: sdeconv.deconv

.. autosummary::
    :toctree: generated
    :nosignatures:

    SWiener
    SRichardsonLucy
    Spitfire
    Noise2VoidDeconv
    SelfSupervisedNNDeconv
    NNDeconv


Interfaces
----------

Available interfaces to create a new PSF generator or a new deconvolution algorithm are:

.. list-table:: Interfaces
   :widths: 25 75

   * - :class:`SPSFGenerator <sdeconv.psfs.interface.SPSFGenerator>`
     - Interface for creating a new PSF generator
   * - :class:`SDeconvFilter <sdeconv.deconv.interface.SDeconvFilter>`
     - Interface for creating a deconvolution filter that does not need neural network
   * - :class:`NNModule <sdeconv.deconv.interface_nn.NNModule>`
     - Interface for creating a deconvolution filter using a neural network
