/// \file SNormalize.cpp
/// \brief SNormalize class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include "math.h"

#include "SNormalize.h"
#include "SException.h"

#ifdef SL_USE_OPENMP
#include "omp.h"
#endif

namespace SImg{

std::string SNormalize::Max = "max";
std::string SNormalize::Sum = "sum";
std::string SNormalize::L2 = "L2";
std::string SNormalize::RC = "rc";
std::string SNormalize::Bits8 = "8bits";
std::string SNormalize::Bits12 = "12bits";
std::string SNormalize::Bits16 = "16bits";

void normalize(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output, std::string method)
{
    if ( method == SNormalize::Max ){
        normMinMax(image, sx, sy, sz, st, sc, output);
    }
    else if ( method == SNormalize::Sum ){
        normSum(image, sx, sy, sz, st, sc, output);
    }
    else if ( method == SNormalize::L2 ){
        normL2(image, sx, sy, sz, st, sc, output);
    }
    else if ( method == SNormalize::RC ){
        normRC(image, sx, sy, sz, st, sc, output);
    }
    else if ( method == SNormalize::Bits8 ){
        normValue(image, sx, sy, sz, st, sc, output, 255.0);
    }
    else if ( method == SNormalize::Bits12 ){
        normValue(image, sx, sy, sz, st, sc, output, pow(2, 12)-1 );
    }
    else if ( method == SNormalize::Bits16 ){
        normValue(image, sx, sy, sz, st, sc, output, pow(2, 16)-1 );
    }
}

void normMinMax(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output)
{
    // calculate min and max
    int bs = sx*sy*sz*st*sc;
    float max = image[0];
    float min = image[0];
    for (unsigned int i = 1 ; i < bs ; i++){
        if (image[i] > max){
            max = image[i];
        }
        if (image[i] < min){
            min = image[i];
        }
    }
    float invMaxMenusMin = 1.0 / (max - min);

    // normalize
    //output = new float[bs];
#pragma omp parallel for
    for (int i = 0 ; i < bs ; i++){
        output[i] = (image[i]-min) * invMaxMenusMin;
    }
}

void normSum(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output)
{
    // calculate sum
    int bs = sx*sy*sz*st*sc;
    float sum = 0.0;
    for (unsigned int i = 0 ; i < bs ; i++){
        sum += image[i];
    }

    // normalize
    //output = new float[bs];
#pragma omp parallel for
    for (int i = 0 ; i < bs ; i++){
        output[i] = (image[i]) / sum;
    }
}

void normL2(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output)
{
    // calculate normL2
    int bs = sx*sy*sz*st*sc;
    float norm = 0.0;
    for (int i = 0 ; i < bs ; i++){
        norm += image[i]*image[i];
    }
    norm = sqrt(norm);

    // normalize
    //output = new float[bs];
#pragma omp parallel for
    for (int i = 0 ; i < bs ; i++){
        output[i] = image[i] / norm;
    }
}

void normRC(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output)
{
    // calculate normL2
    int n = sx*sy*sz*st*sc;
    float mean = 0.0;
    float var = 0.0;
    float v;
    for (int i = 1 ; i < n; i++){
        v = image[i];
        mean += v;
        var += v*v;
    }

    mean /=  n;
    var = sqrt((var-mean*mean/n)/(n-1));

    // normalize
    //output = new float[n];
#pragma omp parallel for
    for (int i = 0 ; i < n ; i++){
        output[i] = (image[i]-mean) / var;
    }
}

void normValue(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output, float normValue)
{
    // normalize
    int bs = sx*sy*sz*st*sc;
    //output = new float[bs];
#pragma omp parallel for
    for (int i = 0 ; i < bs ; i++){
        output[i] = image[i] / normValue;
    }
}

}