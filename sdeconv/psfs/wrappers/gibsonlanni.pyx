include "gibsonlanni.pxi"

import  numpy as np
cimport numpy as np

np.import_array()

def py_gibson_lanni_psf(int sx, int sy, int sz, float res_lateral, float res_axial,
                        float numerical_aperture, float lambd, float ti0, float ni,
                        float ns):

    cdef np.ndarray[np.float32_t, ndim=3, mode='c'] im_out = np.zeros((sx, sy, sz), dtype=np.float32)

    cdef float particle_axial_position = 0
    cdef float ng = 1.5
    cdef float ng0 = 1.5
    cdef float ni0 = ni
    gibson_lanni_psf(<float*> im_out.data, sx, sy, sz,
                         res_lateral, res_axial, numerical_aperture, lambd,
                         ti0, ni0, ni, ng0, ng, ns, particle_axial_position)
    return im_out
