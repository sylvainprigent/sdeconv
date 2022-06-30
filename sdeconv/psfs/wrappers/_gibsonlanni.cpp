/// \file sgibsonlannipsf.cpp
/// \brief sgibsonlannipsf definitions
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include "_gibsonlanni.h"

#include "math.h"
#include <iostream>


    void gibson_lanni_psf(float* buffer_out, unsigned int sx, unsigned int sy, unsigned int sz,
                          const float& res_lateral, const float& res_axial, const float& numerical_aperture, const float& lambda,
                          const float& ti0, const float& ni0, const float& ni, const float& ng0, const float&ng, const float& ns, const float& particle_axial_position)
    {

		for (int i = 0 ; i < sx*sy*sz ; ++i)
		{
			buffer_out[i] = 0.0;
		}

        // The center of the image in units of [pixels]
		float x0 = (float(sx) - 1) / 2.0;
		float y0 = (float(sy) - 1) / 2.0;

        // Lateral particle position in units of [pixels]
		float xp = x0;
		float yp = y0;
        int OVER_SAMPLING = 2;
		float ti0_scaled = ti0 * 1e-6;
		float particle_axial_position_scaled = particle_axial_position * 1e-9;
		float lambda_scaled = lambda*1e-9;

        for (unsigned int z = 0 ; z < sz ; ++z)
        {
            float param_ti = ti0_scaled + res_axial * 1e-9 * (z - (sz - 1.0) / 2.0);

			// Radial locations
			int maxRadius = ((int) floor(sqrt((sx - x0) * (sx - x0) + (sy - y0) * (sy - y0))+0.5)) + 1;

            int r_length = maxRadius * OVER_SAMPLING;
			float* r = new float[r_length];
			float* h = new float[r_length];

            #pragma omp parallel for
			for (int n = 0; n < r_length; n++) {
				r[n] = ((float) n) / ((float) OVER_SAMPLING);
				h[n] = kirchhoff_diffraction_simpson(r[n] * res_lateral * 1e-9, numerical_aperture, lambda_scaled,
                                                     ns, ni, ti0_scaled, param_ti, particle_axial_position_scaled);
			}
			// Linear interpolation of the pixels values
			float rPixel, value;
			int index;
			#pragma omp parallel for
			for (int x = 0; x < sx; x++) {
				for (int y = 0; y < sy; y++) {
					rPixel = sqrt((x - xp) * (x - xp) + (y - yp) * (y - yp));
					index = (int) floor(rPixel * OVER_SAMPLING);
					value = h[index] + (h[index + 1] - h[index]) * (rPixel - r[index]) * OVER_SAMPLING;
					buffer_out[z +sz*(y + sy*x)] = value;
				}
			}
        }

    }

    /// \brief Simpson approximation for the Kirchhoff diffraction integral
    /// \param[in] r Radial distance of the detector relative to the optical axis
    float kirchhoff_diffraction_simpson(float r, const float& numerical_aperture, const float& lambda,
                                        const float& p_ns, const float& p_ni,
                                        const float& p_ti0, const float& p_ti, const float& p_particleAxialPosition)
    {
        float TOL = 1e-1;
        int K = 6;
        float a = 0.0; // Lower and upper limits of the integral
        float b = p_ns / numerical_aperture;
        if (b > 1)
        {
            b = 1.0;
        }
		int N; // number of sub-intervals
		int k; // number of consecutive successful approximations
		float del; // integration interval
		int iteration; // number of iterations.
		float curDifference; // Stopping criterion

		float realSum, imagSum, rho;
		float* sumOddIndex = new float[2];
        float* sumEvenIndex = new float[2];
		float* valueX0 = new float[2];
        float* valueXn = new float[2];
		float* value = new float[2];

		float curI = 0.0, prevI = 0.0;

		// Initialization of the Simpson sum (first iteration)
		N = 2;
		del = (b - a) / 2.0;
		k = 0;
		iteration = 1;
		rho = (b - a) / 2.0;
		sumOddIndex = integrand(rho, r, lambda, numerical_aperture, p_ns, p_ni, p_ti0, p_ti, p_particleAxialPosition);
		sumEvenIndex[0] = 0.0;
		sumEvenIndex[1] = 0.0;

		valueX0 = integrand(a, r, lambda, numerical_aperture, p_ns, p_ni, p_ti0, p_ti, p_particleAxialPosition);
		valueXn = integrand(b, r, lambda, numerical_aperture, p_ns, p_ni, p_ti0, p_ti, p_particleAxialPosition);

		realSum = valueX0[0] + 2.0 * sumEvenIndex[0] + 4.0 * sumOddIndex[0] + valueXn[0];
		imagSum = valueX0[1] + 2.0 * sumEvenIndex[1] + 4.0 * sumOddIndex[1] + valueXn[1];
		curI = (realSum * realSum + imagSum * imagSum) * del * del;

		prevI = curI;
		curDifference = TOL;

		// Finer sampling grid until we meet the TOL value with the specified
		// number of repetitions, K
		while (k < K && iteration < 10000) {
			iteration++;
			N *= 2;
			del = del / 2.0;
			sumEvenIndex[0] = sumEvenIndex[0] + sumOddIndex[0];
			sumEvenIndex[1] = sumEvenIndex[1] + sumOddIndex[1];
			sumOddIndex[0] = 0.0;
			sumOddIndex[1] = 0.0;
			for (int n = 1; n < N; n = n + 2) {
				rho = n * del;
				value = integrand(rho, r, lambda, numerical_aperture, p_ns, p_ni, p_ti0, p_ti, p_particleAxialPosition);
				sumOddIndex[0] += value[0];
				sumOddIndex[1] += value[1];
			}
			realSum = valueX0[0] + 2.0 * sumEvenIndex[0] + 4.0 * sumOddIndex[0] + valueXn[0];
			imagSum = valueX0[1] + 2.0 * sumEvenIndex[1] + 4.0 * sumOddIndex[1] + valueXn[1];
			curI = (realSum * realSum + imagSum * imagSum) * del * del;

			// Relative error between consecutive approximations
			if (prevI == 0.0)
				curDifference = fabs((prevI - curI) / 1e-5);
			else
				curDifference = fabs((prevI - curI) / curI);

			if (curDifference != curDifference || curDifference <= TOL)
				k++;
			else
				k = 0;

			prevI = curI;
		}
		if (curI != curI){
			curI = 0.0;
		}
		return curI;
    }

    float* integrand(float rho, float r, const float& lambda, const float& NA, const float& p_ns, const float& p_ni,
                     const float& p_ti0, const float& p_ti, const float& p_particleAxialPosition)
    {

		// 'rho' is the integration parameter.
		// 'r' is the radial distance of the detector relative to the optical
		// axis.
		// NA is assumed to be less than 1.0, i.e. it assumed to be already
		// normalized by the refractive index of the immersion layer, ni.
		// The return value is a complex number.

		float k0 = 2 * 3.141592653589793238463 / lambda;
		float besselValue = bessel_j0(k0 * NA * r * rho);

		float OPD, OPD1, OPD3; // Optical path differences
		float* I = new float[2];

		if ((NA * rho / p_ns) > 1){
            std::cout << "p.NA*rho/p.ns is bigger than 1. (ns,NA,rho)=(" << p_ns << ", " << NA << ", " << rho << std::endl;
        }

		// Saving some computation time
		OPD1 = p_ns * p_particleAxialPosition * sqrt(1 - (NA * rho / p_ns) * (NA * rho / p_ns));
		OPD3 = p_ni * (p_ti - p_ti0) * sqrt(1 - (NA * rho / p_ni) * (NA * rho / p_ni));
		OPD = OPD1 + OPD3;

		float W = k0 * OPD;

		// The real part
		I[0] = besselValue * cos(W) * rho;
		// The imaginary part
		I[1] = besselValue * sin(W) * rho;

		return I;
	}

    float bessel_j0(float x)
    {
        double ax,z;
        double xx,y,ans,ans1,ans2;
        if ((ax=fabs(x)) < 8.0) {
            y=x*x;
            ans1 = 57568490574.0+y*(-13362590354.0+y*(651619640.7
                   +y*(-11214424.18+y*(77392.33017+y*(-184.9052456)))));
            ans2 = 57568490411.0+y*(1029532985.0+y*(9494680.718
                   +y*(59272.64853+y*(267.8532712+y*1.0))));
            ans=ans1/ans2;
        }
        else{
			z=8.0/x;
			y=z*z;
			xx=x-0.785398164;
            ans1 = 1.0+y*(-0.1098628627e-2+y*(0.2734510407e-4
                   +y*(-0.2073370639e-5+y*0.2093887211e-6)));
            ans2 = -0.1562499995e-1+y*(0.1430488765e-3
                   +y*(-0.6911147651e-5+y*(0.7621095161e-6
                   -y*0.934945152e-7)));
            ans=sqrt(0.636619772/ax)*(cos(xx)*ans1-z*sin(xx)*ans2);
        }
        return float(ans);
    }
