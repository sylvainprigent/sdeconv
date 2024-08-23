# SDeconv

**SDeconv** is a python framework to develop scientific image deconvolution algorithms. This 
library has been developed for microscopy 2D and 3D images, but can be use to any image 
deconvolution application.

# System Requirements

## Software Requirements

### OS Requirements

The `SDeconv` development version is tested on *Windows 10*, *MacOS* and *Linux* operating systems. 
The developmental version of the package has been tested on the following systems:

- Linux: 20.04.4 
- Mac OSX: Mac OS Catalina 10.15.7    
- Windows: 10 

# install

## Library installation from PyPI

1. Install an [Anaconda](https://www.anaconda.com/download/) distribution of Python -- Choose **Python 3.9** and your operating system. Note you might need to use an anaconda prompt if you did not add anaconda to the path.
2. Open an anaconda prompt / command prompt with `conda` for **python 3** in the path
3. Create a new environment with `conda create --name sdeconv python=3.9`.
4. To activate this new environment, run `conda activate sdeconv`
5. To install the `SDeconv`library, run `python -m pip install sdeconv`. 

if you need to update to a new release, use:
~~~sh
python -m pip install sdeconv --upgrade
~~~

## Library installation from source

This installation is for developers or people who want the last features in the ``main`` branch.

1. Install an [Anaconda](https://www.anaconda.com/download/) distribution of Python -- Choose **Python 3.9** and your operating system. Note you might need to use an anaconda prompt if you did not add anaconda to the path.
2. Open an anaconda prompt / command prompt with `conda` for **python 3** in the path
3. Create a new environment with `conda create --name sdeconv python=3.9`.
4. To activate this new environment, run `conda activate sdeconv`
5. Pull the source code from git with `git pull https://github.com/sylvainprigent/sdeconv.git 
6. Then install the `SDeconv` library from you local dir with: `python -m pip install -e ./sdeconv`. 

## Use SDeconv with napari

The SDeconv library is embedded in a napari plugin that allows using ``SDeconv`` with a graphical interface.
Please refer to the [`SDeconv` napari plugin](https://www.napari-hub.org/plugins/napari-sdeconv) documentation to install and use it.

# SDeconv documentation

The full documentation with tutorial and docstring is available [here](https://sylvainprigent.github.io/sdeconv/)
