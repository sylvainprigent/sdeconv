/// \file sl_utils.h
/// \brief sl_utils definitions
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#ifndef sl_utils_H
#define sl_utils_H

#include <string>

namespace SImg{

void normalize_intensities(float* buffer, unsigned int buffer_length, std::string method = "max");
void laplacian_2d(float* buffer_in, float* buffer_out, unsigned int sx, unsigned int sy, int connectivity);

}
#endif /* !sl_utils_H */