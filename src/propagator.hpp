/*
 * Copyright (c) 2026 Ugwu Valentine Nnannaya.
 * All rights reserved.
 *
 * This project is proprietary. Unauthorized copying,
 * modification, or distribution is not permitted.
 */

#ifndef PROPAGATOR_HPP
#define PROPAGATOR_HPP

#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

#include <logicnn.h>
#include <logicnn_backprop.h>

#include "loss.hpp"
#include "dataset.hpp"
#include "neural.hpp"

class propagator
{

private:

	float * const params_;

	size_t& rem_;
	float& mse_sum_;
	float& mae_sum_;
	float& cce_sum_;
	float& acc_sum_;
	std::thread thread_;
	std::mutex& mtx_;
	std::condition_variable& work_cv_;
	std::condition_variable& done_cv_;
	batch& batch_;
    bool& batch_ready_;
	bool& stop_;

	forward_pass fpass_;
	backward_pass bpass_;

private:

	void iterator();

public:

	~propagator();

	propagator(
			parameters& params
		,   size_t& rem
		,   float& mse_sum
		,   float& mae_sum
		,   float& cce_sum
		,   float& acc_sum
		,   std::mutex& mtx
		,   std::condition_variable& work_cv
		,   std::condition_variable& done_cv
		,   batch& batch
        ,   bool& batch_ready
		,   bool& stop
	);

};

#endif // PROPAGATOR_HPP