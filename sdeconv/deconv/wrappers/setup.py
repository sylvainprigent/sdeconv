import os
from os.path import join
import numpy


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('wrappers', parent_package, top_path)

    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    config.add_extension('_spitfire_deconv',
                         sources=['_spitfire_deconv.pyx',
                                  'SObserver.cpp', 'SObservable.cpp',
                                  'SObserverConsole.cpp',
                                  'SShift.cpp', 'SMath.cpp', 'SNormalize.cpp',
                                  'spitfire2d.cpp', 'spitfire3d.cpp',
                                  'SFFT.cpp'],
                         include_dirs=[numpy.get_include(), os.environ['FFT_INCLUDE']],
                         libraries=['fftw3f'] + libraries,
                         language='c++',
                         extra_link_args=['-lstdc++'],
                         extra_compile_args=['-std=c++11', '-v']
                         )

    config.add_extension('_richardson_lucy_deconv',
                         sources=['_richardson_lucy_deconv.pyx',
                                  'SFFT.cpp', 
                                  'SObserver.cpp', 'SObservable.cpp',
                                  'srichardsonlucy.cpp', 'SShift.cpp',
                                  'SObserverConsole.cpp'],
                         include_dirs=[numpy.get_include(), os.environ['FFT_INCLUDE']],
                         libraries=['fftw3f'] + libraries,
                         language='c++',
                         extra_link_args=['-lstdc++'],
                         extra_compile_args=['-std=c++11', '-v']
                         )

    config.add_extension('_wiener_deconv',
                         sources=['_wiener_deconv.pyx',
                                  'SFFT.cpp', 'sutils.cpp',
                                  'SObserver.cpp', 'SObservable.cpp',
                                  'swiener.cpp', 'SShift.cpp'],
                         include_dirs=[numpy.get_include(), os.environ['FFT_INCLUDE']],
                         libraries=['fftw3f'] + libraries,
                         language='c++',
                         extra_link_args=['-lstdc++'],
                         extra_compile_args=['-std=c++11', '-v']
                         )

    config.add_extension('_psfs',
                         sources=['_psfs.pyx', 
                                  'sgibsonlannipsf.cpp', 'sgaussianpsf.cpp', 
                                  'SObserver.cpp', 'SObserverConsole.cpp'],
                         include_dirs=[numpy.get_include(), os.environ['FFT_INCLUDE']],
                         libraries=['fftw3f'] + libraries,
                         language='c++',
                         extra_link_args=['-lstdc++'],
                         extra_compile_args=['-std=c++11', '-v']
                         )

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
