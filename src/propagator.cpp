/*
 * Copyright (c) 2026 Ugwu Valentine Nnannaya.
 * All rights reserved.
 *
 * This project is proprietary. Unauthorized copying,
 * modification, or distribution is not permitted.
 */

#include <propagator.hpp>
#include <loss.hpp>

void propagator::iterator()
{
	for (;;) {

		std::unique_lock<std::mutex> lock(mtx_);

		cv_.wait(lock, [this]() {
			return stop_ || !batch_.empty();
		});

		if (batch_.empty()) 
		{
			if (stop_) break;
			continue; 
		}

		auto data = data_.front();

		data_.pop();
		lock.unlock();
		cv_.notify_one();

		auto x = data_.sample.input();
		auto y = data_.sample.output();

		auto y_hat = fpass_(x);

		auto error = loss::forward(y_hat, y);
		auto delta = loss::backward(y_hat, y);

		mse_sum_ += error.mse;
		cce_sum_ += error.cce;
		acc_sum_ += error.accuracy;

		bpass_(delta, data_.grad);

		it_++;
	}
}

propagator::~propagator()
{
	thread_.join();
}

propagator::propagator
	(
			const float * params
		,   std::atomic<size_t>& it
		,   std::atomic<float>& mse_sum
		,   std::atomic<float>& cce_sum
		,   std::atomic<float>& acc_sum
		,   std::mutex& mtx
		,   std::condition_variable& cv
		,   batch& batch
		,   bool& stop
	)
		:   params_(params)
		,   thread_(propagator::iterator, this)
		,   mtx_(mtx)
		,   cv_(cv)
		,   batch_(batch)
		,   stop_(stop)
		,   fpass_(params)
		,   bpass_(fpass_)
{}