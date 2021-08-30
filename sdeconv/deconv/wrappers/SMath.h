/// \file SMath.h
/// \brief SMath
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

/// \class SMath
/// \brief class defining constant values and basic math
class SMath
{
public:
    static const float PI;
    static const float EPSILON;
    static const float FMAX;
    static const float FMIN;

public:
    static float min(float x, float y);
    static float max(float x, float y);
    static float median(float* data, unsigned int length);
    static int* sortShell(float* vectIn, float* vectOut, unsigned int length);
};
