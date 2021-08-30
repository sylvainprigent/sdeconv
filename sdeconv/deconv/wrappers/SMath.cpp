/// \file SMath.cpp
/// \brief SMath
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2019

#include "SMath.h"
#include "float.h"
#include "math.h"

const float SMath::PI = float(3.1415927);
const float SMath::EPSILON = float(1e-10);
const float SMath::FMAX = float(DBL_MAX);
const float SMath::FMIN = float(DBL_MIN);

float SMath::min(float x, float y){
    return x<y ? x : y;
}

float SMath::max(float x, float y){
    return x>y ? x : y;
}
