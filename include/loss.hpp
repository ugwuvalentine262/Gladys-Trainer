/*
 * Copyright (c) 2026 Ugwu Valentine Nnannaya.
 * All rights reserved.
 *
 * This project is proprietary. Unauthorized copying,
 * modification, or distribution is not permitted.
 */

#ifndef LOSS_HPP
#define LOSS_HPP

#include <neural.hpp>

struct error
{
	error(
			float mse_
		,   float cce_
		,   float accuracy_
	)
		: mse(mse_)
		, cce(cce_)
		, accuracy(accuracy_)
	{}

public:

    const float mse;
	const float cce;
	const float accuracy;
};

struct loss
{
	static value forward(const neural_output& y_hat, const neural_output& y);

	static neural_output backward(const neural_output& y_hat, const neural_output& y);
};

#endif // LOSS_HPP