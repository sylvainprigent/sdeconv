/// \file SShift.h
/// \brief SShift functions
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

namespace SImg{

void shift2D(float* buffer_in, float* buffer_out, unsigned int sx, unsigned int sy, const int& shift_x, const int& shift_y);
void shift3D(float *buffer_in, float *buffer_out, unsigned int sx, unsigned int sy, unsigned int sz, const int &shift_x, const int &shift_y, const int &shift_z);
    
void shift(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output, const int& m_shiftX, const int& m_shiftY, const int& m_shiftZ);

}