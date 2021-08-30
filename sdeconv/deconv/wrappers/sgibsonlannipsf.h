/// \file sgibsonlannipsf.h
/// \brief sgibsonlannipsf definitions
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#ifndef sgibsonlannipsf_H
#define sgibsonlannipsf_H

namespace SImg{

    /// \brief Generate a PSF with the Gibson Lanni model (adapted from Daniel Sage Fiji plugin)
    /// \param[out] buffer_out Buffer of the generated PSF image
    /// \param[in] sx Number of rows in the PSF image (must be >= 3)
    /// \param[in] sy Number of columns in the PSF image (must be >= 4)
    /// \param[in] sz Number of slices in the PSF image (must be >= 4)
    /// \param[in] ti0 Working distance of the objective (design value). This is also the width of the immersion layer
    /// \param[in] ti Working distance of the objective (experimental value). influenced by the stage displacement
    /// \param[in] ni0 Immersion medium refractive index (design value)
    /// \param[in] ni Immersion medium refractive index (experimental value) 
    /// \param[in] tg0 Coverslip thickness (design value)
    /// \param[in] tg Coverslip thickness (experimental value)
    /// \param[in] ng0 Coverslip refractive index (design value) 
    /// \param[in] ng Coverslip refractive index (experimental value)
    /// \param[in] ns Sample refractive index
    /// \param[in] res_lateral Lateral resolution
    /// \param[in] res_axial Axial resolution
    /// \param[in] numerical_aperture Numerical aperture
    /// \param[in] particle_axial_position Axial position of the particle 
    /// \param[in] lambda Wavelength
    void gibson_lanni_psf(float* buffer_out, unsigned int sx, unsigned int sy, unsigned int sz, 
                          const float& res_lateral, const float& res_axial, const float& numerical_aperture, const float& lambda,
                          const float& ti0, const float& ni0, const float& ni, const float& ng0, const float&ng, const float& ns, const float& particle_axial_position);

    float kirchhoff_diffraction_simpson(float r, const float& numerical_aperture, const float& lambda, 
                                        const float& p_ns, const float& p_ni, 
                                        const float& p_ti0, const float& p_ti, const float& p_particleAxialPosition);
    float* integrand(float rho, float r, const float& lambda, const float& NA, const float& p_ns, const float& p_ni, 
                     const float& p_ti0, const float& p_ti, const float& p_particleAxialPosition);   
    float bessel_j0(float x);              

}



#endif /* !sgibsonlannipsf_H */
