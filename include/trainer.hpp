/*
 * Copyright (c) 2026 Ugwu Valentine Nnannaya.
 * All rights reserved.
 *
 * This project is proprietary. Unauthorized copying,
 * modification, or distribution is not permitted.
 */

#ifndef TRAINER_HPP
#define TRAINER_HPP

#include <string>
#include <chrono>
#include <cstdint>

#include "adam.hpp"
#include "propagator.hpp"
#include "dataset.hpp"
#include "neural.hpp"
#include "summary.hpp"
#include "adam.hpp"

class trainer
{
	using propagators=std::vector<propagator>;

private:

	parameters params_;
	gradients grad_;
	float xgrad_[BATCH_SIZE][PARAM_COUNT + CACHE_LINE];
	propagators props_;
	dataset dataset_;
	adam adam_;
	std::atomic<float> mse_sum_;
	std::atomic<float> cce_sum_;
	std::atomic<float> acc_sum_;

public:

	summary train();

	trainer();

};

#endif // TRAINER_HPP