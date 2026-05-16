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
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>

#include <logicnn.h>
#include <logicnn_backprop.h>

#include "cce.hpp"
#include "mse.hpp"
#include "dataset.hpp"
#include "neural.hpp"

class propagator
{

public:

	struct input_data 
	{
		const sample& sample;
		float * const grad;

	public:

		input_data(const sample&, float[]);
	};

	using batch=std::queue<input_data>;

private:

	const float * params_;

    std::atomic<size_t>& it_;
	std::atomic<float>& mse_sum_;
	std::atomic<float>& cce_sum_;
	std::atomic<float>& acc_sum_;
	std::thread thread_;
	std::mutex& mtx_;
	std::condition_variable& cv_;
	batch& batch_;
	bool& stop_;

	forward_pass fpass_;
	backward_pass bpass_;

private:

	void iterator();

public:

	~propagator();

	propagator(
			const float * params
		,   std::atomic<size_t>& it
		,   std::atomic<float>& mse_sum
		,   std::atomic<float>& cce_sum
		,   std::atomic<float>& acc_sum
		,   std::mutex& mtx
		,   std::condition_variable& cv
		,   batch& batch
		,   bool& stop
	);

};

#endif // PROPAGATOR_HPP