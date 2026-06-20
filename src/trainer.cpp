/*
 * Copyright (c) 2026 Ugwu Valentine Nnannaya.
 * All rights reserved.
 *
 * This project is proprietary. Unauthorized copying,
 * modification, or distribution is not permitted.
 */

#include <chrono>
#include <Eigen/Dense>
#include <fstream>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <random>
#include <string>
#include <deque>
#include <queue>

#include "io.hpp"
#include "trainer.hpp"

void trainer::train()
{
    using clock = std::chrono::steady_clock;

	auto start = clock::now();

    dataset_.shuffle();

	mse_sum_=cce_sum_=acc_sum_=0;

	for (const auto& sample : dataset_)
	{    
        batch_.emplace(sample, xgrad_.at(batch_.size()));

		if (batch_.size()==batch_size_)
		{
            batch_ready_=true;
            rem_=batch_.size();

            work_cv_.notify_all();
            std::unique_lock<std::mutex> lock(mtx_);
			done_cv_.wait(lock, [this]() { return !batch_ready_ && !rem_; });

			Eigen::Map<Eigen::VectorXf> v(grad_.data, PARAM_COUNT);

			for (size_t i=0; i < batch_size_; i++)
			{
				Eigen::Map<const Eigen::VectorXf> h(xgrad_.at(i).data, PARAM_COUNT);

				v += h;
			}

			v.array() /= batch_size_;

			adam_.step();
			adam_.zero_grad();

            for (const auto& p : params_.data)
            {
		        if (std::isnan(p) || std::isinf(p))
                {
                    log_ 
                        << "Undefined neural network parameters."
                        << std::endl;

                    std::exit(EXIT_FAILURE);
                }
            }

		}
	}

	auto end = clock::now();

	auto mse_error = mse_sum_ / dataset_.size();
	auto cce_error = cce_sum_ / dataset_.size();
	auto accuracy = acc_sum_ / dataset_.size();

	auto seconds = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();

	struct { int hr, min, sec; } elapsed;

	elapsed.hr = seconds / 3600;
	elapsed.min = (seconds % 3600) / 60;
	elapsed.sec = seconds % 60;

	log_ 
		<< std::fixed 
		<< std::setprecision(6)
		<< "mse: " 
		<< std::setw(8)
		<< mse_error
		<< " | cce: "
		<< std::setw(8)
		<< cce_error
		<< " | accuracy: "
		<< std::setw(8) 
		<< accuracy
		<< " | elapsed: "
		<< std::setfill('0')
		<< std::setw(2)
		<< elapsed.hr 
		<< ":"
		<< std::setw(2)
		<< elapsed.min
		<< ":"
		<< std::setw(2)
		<< elapsed.sec
        << std::endl;

    std::ofstream file(NNFILE, std::ios::binary);

    if (!file) {

		log_ 
	        << "Unable to save neural network to file."
			<< std::endl;

	    std::exit(EXIT_FAILURE);
	}

	write(file, params_.data, PARAM_COUNT);
	adam_.save();
}

trainer::~trainer()
{
	{
		std::lock_guard<std::mutex> lock(mtx_);
		stop_=true;
	}

	work_cv_.notify_all();
}

trainer::trainer(
			size_t workers
		,   size_t batch_size
		,   float alpha
		,   std::ofstream& log
	)
		:   log_(log)
		,   params_ {}
		,   grad_ {}
		,   xgrad_(batch_size)
		,   props_ {}
		,   dataset_(log)
		,   adam_(params_.data, grad_.data, alpha)
		,   batch_size_(batch_size)
		,   rem_(0)
		,   mse_sum_(0)
		,   cce_sum_(0)
		,   acc_sum_(0)
		,   batch_ {}
        ,   batch_ready_(false)
		,   stop_(false)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dist(-0.5, 0.5);

	for (size_t i=0; i < workers; i++)
	{
		props_.emplace_back(

				params_
			,   rem_
			,   mse_sum_
			,   cce_sum_
			,   acc_sum_
			,   mtx_
			,   work_cv_
            ,   done_cv_
			,   batch_
            ,   batch_ready_
			,   stop_
		);
	}

	if (!adam_.load()) {
		adam_.reset();
	}

	std::ifstream ifile(NNFILE, std::ios::binary);

	if (ifile) {
		read(ifile, params_.data, PARAM_COUNT);
	}

	if (!ifile || ifile.eof())
	{
        std::ofstream ofile(NNFILE, std::ios::binary);

		auto bias=params_.data+WDL_BIAS_OFFSET;

		for (int i=0; i < PARAM_COUNT; i++)
		{
		    params_.data[i] = dist(gen);
		}
		bias[0]=bias[1]=bias[2]=0;

        log.close();
        log.open(LOGFILE);
		log
			<< "Initialized "
			<< PARAM_COUNT 
			<< " parameters of neural network.\n"
			<< std::endl;

        if (!ofile) {

		    log 
	            << "Unable to save neural network to file."
			    << std::endl;

	        std::exit(EXIT_FAILURE);
	    }

	    write(ofile, params_.data, PARAM_COUNT);
	}
}

int main(int argc, char *argv[])
{
	auto epochs=10;
	auto batch=50;
	auto alpha=0.001f;
	auto workers=4u;
    auto break_duration=0;

	std::queue<std::string> args(
			std::deque<std::string>(argv + 1, argv + argc)
		);

	std::ofstream log(LOGFILE, std::ios::app);

	if (!log) std::exit(EXIT_FAILURE);

	while (!args.empty())
	{
		auto arg = args.front();

		args.pop();

		if (args.empty()) break;

		auto value = args.front();

		float x=0;

		args.pop();

		try {
			x = std::stof(value);
		}
		catch (const std::invalid_argument& e) {
			log << value << " is not a number\n";
			continue;
		}
		catch (const std::out_of_range& e) {
			log << value << " is too large\n";
			continue;
		}

		if (arg=="epochs") {
			epochs= static_cast<int>(x);
		}
		else if (arg=="batch") {
			batch= static_cast<int>(x);
		}
		else if (arg=="alpha") {
			alpha= static_cast<float>(x);
		}
		else if (arg=="workers") {
			workers= static_cast<unsigned>(x);
		}
        else if (arg=="break") {
            break_duration =static_cast<unsigned>(x);
        }
		else {
			log << "Unknown parameter: " << arg << std::endl;
		}
	}

	trainer trainer(workers, batch, alpha, log);

	for (auto i=0; i< epochs; i++) 
	{
		trainer.train();

        std::this_thread::sleep_for(std::chrono::seconds(break_duration));
	}
}