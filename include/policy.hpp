/*
 * Copyright (c) 2026 Ugwu Valentine Nnannaya.
 * All rights reserved.
 *
 * This project is proprietary. Unauthorized copying,
 * modification, or distribution is not permitted.
 */

#ifndef POLICY_HPP
#define POLICY_HPP

#include <array>

#define POLICY_DIM 1024

using logits=std::array<float, POLICY_DIM>;
using policy=std::array<float, POLICY_DIM>;

struct move
{
	int from;
	int to;
	int piece;
	int capt;
	int promo;
	bool null;
};

size_t policy_map_index(const move& move);

policy softmax(const logits& z);

bool accurate(const logits& z, const policy& y);

#endif // POLICY_HPP