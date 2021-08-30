include "_wiener_deconv.pxi"

import  numpy as np
cimport numpy as np

np.import_array()

def py_wiener_deconv_2d(np.ndarray[np.float32_t, ndim=2, mode='c'] image_in,
                          np.ndarray[np.float32_t, ndim=2, mode='c'] psf,
                          float lambda_, int connectivity):

    if image_in.ndim != 2:
        raise ValueError('Only 2D images are supported by py_wiener_deconv_2d')

    cdef np.ndarray[np.float32_t, ndim=2, mode='c'] im_out = np.zeros((image_in.shape[0], image_in.shape[1]),
                                                                          dtype=np.float32)

    wiener_deconv_2d(<float*> image_in.data, <float*> psf.data, <float*> im_out.data, image_in.shape[0],
                     image_in.shape[1], lambda_, connectivity)

    return im_out


def py_wiener_deconv_3d(np.ndarray[np.float32_t, ndim=3, mode='c'] image_in,
                          np.ndarray[np.float32_t, ndim=3, mode='c'] psf,
                          float lambda_, int connectivity):

    if image_in.ndim != 3:
        raise ValueError('Only 3D images are supported by py_wiener_deconv_3d')

    cdef np.ndarray[np.float32_t, ndim=3, mode='c'] im_out = np.zeros((image_in.shape[0],
                                                                       image_in.shape[1],
                                                                       image_in.shape[2]),
                                                                          dtype=np.float32)

    wiener_deconv_3d(<float*> image_in.data, <float*> psf.data, <float*> im_out.data,
                     image_in.shape[0], image_in.shape[1], image_in.shape[2],
                     lambda_, connectivity)
    return im_out

