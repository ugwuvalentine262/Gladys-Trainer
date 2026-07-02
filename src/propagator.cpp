/*
 * Copyright (c) 2026 Ugwu Valentine Nnannaya.
 * All rights reserved.
 *
 * This project is proprietary. Unauthorized copying,
 * modification, or distribution is not permitted.
 */

#include <iostream>
#include <cmath>
#include <cassert>
#include <cstdlib>

#include "propagator.hpp"
#include "loss.hpp"

void propagator::iterator()
{
	for (;;) {

        input_data data;

        {
		    std::unique_lock<std::mutex> lock(mtx_);

		    work_cv_.wait(lock, [this]() {
			    return stop_ || (batch_ready_ && !batch_.empty());
		    });

            if (stop_) return;
            if (batch_.empty()) continue;

            data = batch_.front();
		    batch_.pop();
        }

        auto grad = data.grad_->data;

		auto x = data.sample_->input();
		auto y = data.sample_->output();

		auto y_hat = fpass_(x);

		auto error = loss::forward(y_hat, y);
		auto delta = loss::backward(y_hat, y);

		bpass_(delta, grad);

        {
            std::lock_guard<std::mutex> lock(mtx_);

        	mse_sum_ += error.mse;
        	mae_sum_ += error.mae;
		    cce_sum_ += error.cce;
		    acc1_sum_ += error.top_1_accuracy;
		    acc3_sum_ += error.top_3_accuracy;

            if (!--rem_) {
                batch_ready_=false;
                done_cv_.notify_one();
            }
        }	
	}
}

propagator::~propagator()
{
	thread_.join();
}

propagator::propagator
	(
			parameters& params
		,   size_t& rem
		,   float& mse_sum
		,   float& mae_sum
		,   float& cce_sum
		,   float& acc1_sum
		,   float& acc3_sum
		,   std::mutex& mtx
		,   std::condition_variable& work_cv
		,   std::condition_variable& done_cv
		,   batch& batch
        ,   bool& batch_ready
		,   bool& stop
	)
		:   params_(params.data)
        ,   rem_(rem)
        ,   mse_sum_(mse_sum)
        ,   mae_sum_(mae_sum)
        ,   cce_sum_(cce_sum)
        ,   acc1_sum_(acc1_sum)
        ,   acc3_sum_(acc3_sum)
		,   thread_(&propagator::iterator, this)
		,   mtx_(mtx)
		,   work_cv_(work_cv)
		,   done_cv_(done_cv)
		,   batch_(batch)
        ,   batch_ready_(batch_ready)
		,   stop_(stop)
		,   fpass_(params_)
		,   bpass_(fpass_)
{}