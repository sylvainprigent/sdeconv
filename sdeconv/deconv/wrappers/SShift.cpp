/// \file SShift.cpp
/// \brief SShift functions
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2019

#include "SShift.h"

#ifdef SL_USE_OPENMP
#include "omp.h"
#endif

namespace SImg
{

    void shift(float *image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float *output, const int &m_shiftX, const int &m_shiftY, const int &m_shiftZ)
    {
        unsigned long bs = sx * sy * sz * st * sc;
        output = new float[bs];

        int shiftX, shiftY, shiftZ;
        if (sz == 1)
        {
            for (int y = 0; y < sy; y++)
            {
                for (int x = 0; x < sx; x++)
                {

                    shiftX = x + m_shiftX;
                    if (shiftX < 0)
                    {
                        shiftX += sx;
                    }
                    else if (shiftX >= sx)
                    {
                        shiftX -= sx;
                    }

                    shiftY = y + m_shiftY;
                    if (shiftY < 0)
                    {
                        shiftY += sy;
                    }
                    else if (shiftY >= sy)
                    {
                        shiftY -= sy;
                    }

                    output[y + sy * x] = image[shiftY + sy * shiftX];
                }
            }
        }
        else
        {
            for (int z = 0; z < int(sz); z++)
            {
                for (int y = 0; y < int(sy); y++)
                {
                    for (int x = 0; x < int(sx); x++)
                    {

                        shiftX = x + m_shiftX;
                        if (shiftX < 0)
                        {
                            shiftX = shiftX + sx;
                        }
                        else if (shiftX >= int(sx))
                        {
                            shiftX = shiftX - sx;
                        }

                        shiftY = y + m_shiftY;
                        if (shiftY < 0)
                        {
                            shiftY = shiftY + sy;
                        }
                        else if (shiftY >= int(sy))
                        {
                            shiftY = shiftY - sy;
                        }

                        shiftZ = z + m_shiftZ;
                        if (shiftZ < 0)
                        {
                            shiftZ = shiftZ + sz;
                        }
                        else if (shiftZ >= int(sz))
                        {
                            shiftZ = shiftZ - sz;
                        }

                        output[z + sz * (y + sy * x)] = image[shiftZ + sz * (shiftY + sy * shiftX)];
                    }
                }
            }
        }
    }

    void shift3D(float *buffer_in, float *buffer_out, unsigned int sx, unsigned int sy, unsigned int sz, const int &shift_x, const int &shift_y, const int &shift_z)
    {
        int shiftX, shiftY, shiftZ;
        for (int z = 0; z < int(sz); z++)
        {
            for (int y = 0; y < int(sy); y++)
            {
                for (int x = 0; x < int(sx); x++)
                {

                    shiftX = x + shift_x;
                    if (shiftX < 0)
                    {
                        shiftX = shiftX + sx;
                    }
                    else if (shiftX >= int(sx))
                    {
                        shiftX = shiftX - sx;
                    }

                    shiftY = y + shift_y;
                    if (shiftY < 0)
                    {
                        shiftY = shiftY + sy;
                    }
                    else if (shiftY >= int(sy))
                    {
                        shiftY = shiftY - sy;
                    }

                    shiftZ = z + shift_z;
                    if (shiftZ < 0)
                    {
                        shiftZ = shiftZ + sz;
                    }
                    else if (shiftZ >= int(sz))
                    {
                        shiftZ = shiftZ - sz;
                    }

                    buffer_out[z + sz * (y + sy * x)] = buffer_in[shiftZ + sz * (shiftY + sy * shiftX)];
                }
            }
        }
    }

    void shift2D(float *buffer_in, float *buffer_out, unsigned int sx, unsigned int sy, const int &shift_x, const int &shift_y)
    {
        int shiftX, shiftY;
        for (int y = 0; y < sy; y++)
        {
            for (int x = 0; x < sx; x++)
            {

                shiftX = x + shift_x;
                if (shiftX < 0)
                {
                    shiftX += sx;
                }
                else if (shiftX >= sx)
                {
                    shiftX -= sx;
                }

                shiftY = y + shift_y;
                if (shiftY < 0)
                {
                    shiftY += sy;
                }
                else if (shiftY >= sy)
                {
                    shiftY -= sy;
                }

                buffer_out[y + sy * x] = buffer_in[shiftY + sy * shiftX];
            }
        }
    }

}