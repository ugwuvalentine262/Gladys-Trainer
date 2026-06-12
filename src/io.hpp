/*
 * Copyright (c) 2026 Ugwu Valentine Nnannaya.
 * All rights reserved.
 *
 * This project is proprietary. Unauthorized copying,
 * modification, or distribution is not permitted.
 */

#ifndef IO_HPP
#define IO_HPP

#include <cstdint>
#include <bit>
#include <fstream>

inline float reverse_bytes(float f) 
{
    uint32_t x = std::bit_cast<uint32_t>(f);
    x = std::byteswap(x);
    return std::bit_cast<float>(x);
}

inline void write(std::ofstream& file, const float data[], size_t N)
{
    if (std::endian::native == std::endian::little)
    {
        file.write(reinterpret_cast<const char*>(data), sizeof(float) * N);
    }
    else 
    {
        for (size_t i=0; i < N; i++)
        {
            float f = reverse_bytes(data[i]);
            file.write(reinterpret_cast<const char*>(&f), sizeof(float));
        }
    }
}

inline void read(std::ifstream& file, float data[], size_t N)
{
    file.read(reinterpret_cast<char*>(data), sizeof(float) * N);

    if (std::endian::native == std::endian::big)
    {
        for (size_t i=0; i < N; i++)
        {
            data[i] = reverse_bytes(data[i]);
        }
    }
}

#endif // IO_HPP