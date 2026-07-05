/*
 * Copyright (c) 2026 Ugwu Valentine Nnannaya.
 * All rights reserved.
 *
 * This project is proprietary. Unauthorized copying,
 * modification, or distribution is not permitted.
 */

#ifndef LOSS_HPP
#define LOSS_HPP

#include "neural.hpp"

struct error
{
	error(
			float q_mae_
		,   float wdl_cce_
		,   float pi_cce_
		,   float pi_accuracy1_
		,   float pi_accuracy3_
	)
		: q_mae(q_mae_)
        , wdl_cce(wdl_cce_)
		, pi_cce(pi_cce_)
		, pi_accuracy1(pi_accuracy1_)
		, pi_accuracy3(pi_accuracy3_)
	{}

public:

	const float q_mae;
	const float wdl_cce;
	const float pi_cce;
	const float pi_accuracy1;
	const float pi_accuracy3;

};

struct loss
{
	static error forward(const neural_output& y_hat, const neural_output& y);

	static neural_output backward(const neural_output& y_hat, const neural_output& y);

};

#endif // LOSS_HPP