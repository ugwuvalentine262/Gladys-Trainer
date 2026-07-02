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
			float mse_
		,   float mae_
		,   float cce_
		,   float top_1_accuracy_
		,   float top_3_accuracy_
	)
		: mse(mse_)
        , mae(mae_)
		, cce(cce_)
		, top_1_accuracy(top_1_accuracy_)
		, top_3_accuracy(top_3_accuracy_)
	{}

public:

	const float mse;
	const float mae;
	const float cce;
	const float top_1_accuracy;
	const float top_3_accuracy;
};

struct loss
{
	static error forward(const neural_output& y_hat, const neural_output& y);

	static neural_output backward(const neural_output& y_hat, const neural_output& y);

};

#endif // LOSS_HPP