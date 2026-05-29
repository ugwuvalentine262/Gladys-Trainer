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
#include <cstring>
#include <fstream>

inline bool is_little_endian()
{
	uint16_t value = 0x0001U;
	return *reinterpret_cast<uint8_t*>(&value) == 1;
}

inline float reverse_bytes(float f)
{
	uint32_t bits;

	std::memcpy(&bits, &f, sizeof(bits));

	bits =  ((bits & 0x000000FFu) << 24) |
			((bits & 0x0000FF00u) << 8)  |
			((bits & 0x00FF0000u) >> 8)  |
			((bits & 0xFF000000u) >> 24);


	std::memcpy(&f, &bits, sizeof(bits));

	return f;
}

inline void write(std::ofstream& file, const float data[], size_t N)
{
	if (is_little_endian()) {
		file.write(reinterpret_cast<const char*>(data), sizeof(float) * N);
	}
	else {
		for (size_t i=0; i < N; i++)
		{
			float f = reverse_bytes(data[i]);
			reinterpret_cast<const char*>(&f), sizeof(float));
		}
	}
}

inline void read(std::ifstream& file, float data[], size_t N)
{
	file.read(reinterpret_cast<char*>(data), sizeof(float) * N);

	if(!is_little_endian())
	{
		for (size_t i=0; i < N; i++)
		{
			data[i] = reverse_bytes(data[i]);
		}
	}
}

#endif // IO_HPP