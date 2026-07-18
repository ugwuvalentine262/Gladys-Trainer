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

            q_mse_ += error.q_mse;
            q_mae_ += error.q_mae;
            wdl_cce_ += error.wdl_cce;
            pi_cce_ += error.pi_cce;
            pi_accuracy1_ += error.pi_accuracy1;
            pi_accuracy3_ += error.pi_accuracy3;

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
		,   float& q_mse
		,   float& q_mae
		,   float& wdl_cce
		,   float& pi_cce
		,   float& pi_accuracy1
		,   float& pi_accuracy3
		,   std::mutex& mtx
		,   std::condition_variable& work_cv
		,   std::condition_variable& done_cv
		,   batch& batch
        ,   bool& batch_ready
		,   bool& stop
	)
		:   params_(params.data)
        ,   rem_(rem)
	    ,   q_mse_(q_mse)
	    ,   q_mae_(q_mae)
	    ,   wdl_cce_(wdl_cce)
	    ,   pi_cce_(pi_cce)
	    ,   pi_accuracy1_(pi_accuracy1)
	    ,   pi_accuracy3_(pi_accuracy3)
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