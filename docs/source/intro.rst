Introduction
============

SDeconv is a library for 2D and 3D deconvolution of scientific images

Context
-------
SDeconv has been developed in the `Serpico <https://team.inria.fr/serpico/>`_ research team. The goal is to provide a
modular library to perform deconvolution of microscopy images. A classical application of our team is to apply deconvolution in 3D+t
images depecting endosomes with Lattice LightSheet microscopy, and then ease the analysis.

Library components
------------------
SDeconv is written in python3 with pytorch. SDeconv library provides a module for each components of deconvolution algorithms:

* **psfs**: this module defines the interface to implement Point Spread Function generators.
* **deconv**: this module defines the interfaces to implement a deconvolution algorithm with or without neural networks

Furthermore, the library provides sample data, a command line interface, and a application
programing interface to ease the integration of the sdeconv deconvolution algorithms into softwares.
