/*
 * Copyright (c) 2026 Ugwu Valentine Nnannaya.
 * All rights reserved.
 *
 * This project is proprietary. Unauthorized copying,
 * modification, or distribution is not permitted.
 */

#include<Eigen/Dense>

#include <policy.hpp>

bool accurate(const logits& z, const policy& y)
{
	Eigen::Map<const Eigen::VectorXf> z_(z.data(), z.size());
	Eigen::Map<const Eigen::VectorXf> y_(z.data(), y.size());

	Eigen::Index argmax_z=0, argmax_y=0;

	z_.maxCoeff(&argmax_z);
	y_.maxCoeff(&argmax_y);

	return argmax_z==argmax_y;
}