/*
 * Copyright (c) 2026 Ugwu Valentine Nnannaya.
 * All rights reserved.
 *
 * This project is proprietary. Unauthorized copying,
 * modification, or distribution is not permitted.
 */

#include <iostream>
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

    q_mae_=wdl_cce_=pi_cce_=pi_accuracy1_=pi_accuracy3_=0;

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
                    std::cerr
                        << "Undefined neural network parameters."
                        << std::endl;

                    std::exit(EXIT_FAILURE);
                }
            }

		}
	}

	auto end = clock::now();

    q_mae_ /= dataset_.size();
    wdl_cce_ /= dataset_.size();
    pi_cce_ /= dataset_.size();
    pi_accuracy1_ /= dataset_.size();
    pi_accuracy3_ /= dataset_.size();

	auto seconds = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();

	struct { int hr, min, sec; } elapsed;

	elapsed.hr = seconds / 3600;
	elapsed.min = (seconds % 3600) / 60;
	elapsed.sec = seconds % 60;

	std::cout
		<< std::fixed 
		<< std::setprecision(6)
		<< std::setw(10)
		<< q_mae_
		<< " |" 
		<< std::setw(10)
		<< wdl_cce_
		<< " |"
		<< std::setw(10)
		<< pi_cce_
		<< " |"
		<< std::setw(10)
		<< pi_accuracy1_
		<< " |"
		<< std::setw(10)
		<< pi_accuracy3_
        << " |  "
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

		std::cerr 
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
		,   float lambda
	)
		:   params_ {}
		,   grad_ {}
		,   xgrad_(batch_size)
		,   props_ {}
		,   dataset_()
		,   adam_(params_.data, grad_.data, alpha, lambda)
		,   batch_size_(batch_size)
		,   rem_(0)
		,   q_mae_(0)
		,   wdl_cce_(0)
		,   pi_cce_(0)
		,   pi_accuracy1_(0)
		,   pi_accuracy3_(0)
		,   batch_ {}
        ,   batch_ready_(false)
		,   stop_(false)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dist(-RANGE/2.0F, RANGE/2.0F);

	for (size_t i=0; i < workers; i++)
	{
		props_.emplace_back(

				params_
			,   rem_
			,   q_mae_
			,   wdl_cce_
			,   pi_cce_
			,   pi_accuracy1_
			,   pi_accuracy3_
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

		for (int i=0; i < PARAM_COUNT; i++)
		{
		    params_.data[i] = dist(gen);
		}

		std::clog
			<< "Initialized "
			<< PARAM_COUNT 
			<< " parameters of neural network.\n"
			<< std::endl;

        if (!ofile) {
		    std::cerr << "Unable to save neural network to file." << std::endl;
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
    auto lambda=0.0001f;
	auto workers=4u;
    auto break_duration=0;

	std::queue<std::string> args(
			std::deque<std::string>(argv + 1, argv + argc)
		);

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
			std::clog << value << " is not a number\n";
			continue;
		}
		catch (const std::out_of_range& e) {
			std::clog << value << " is too large\n";
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
		else if (arg=="lambda") {
			lambda= static_cast<float>(x);
		}
		else if (arg=="workers") {
			workers= static_cast<unsigned>(x);
		}
        else if (arg=="break") {
            break_duration =static_cast<unsigned>(x);
        }
		else {
			std::clog << "Unknown parameter: " << arg << std::endl;
		}
	}

	trainer trainer(workers, batch, alpha, lambda);

	for (auto i=0; i< epochs; i++) 
	{
        if (!workers) {
            std::cerr << "There are no workers to train the model." << std::endl;
            std::exit(EXIT_FAILURE);
        }
		trainer.train();

        std::this_thread::sleep_for(std::chrono::seconds(break_duration));
	}
}