/*
 * Copyright (c) 2026 Ugwu Valentine Nnannaya.
 * All rights reserved.
 *
 * This project is proprietary. Unauthorized copying,
 * modification, or distribution is not permitted.
 */

#include <fstream>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include <string>

#include "dataset.hpp"
#include "tester.hpp"
#include "io.hpp"
#include "loss.hpp"

void tester::iterator()
{
	forward_pass fpass(params_.data);

	for (;;)
	{
		const sample *sample=nullptr;

		{
			std::lock_guard<std::mutex> lock(mtx_);

			if (stack_.empty())
			{
				return;
			}
			sample = stack_.top();
			stack_.pop();
		}

		auto x = sample->input();
		auto y = sample->output();
		auto y_hat = fpass(x);
		auto error = loss::forward(y_hat, y);

        {
            std::lock_guard<std::mutex> lock(mtx_);

        	mse_sum_ += error.mse;
		    cce_sum_ += error.cce;
		    acc_sum_ += error.accuracy;

            if (--rem_==0) {
                cv_.notify_one();
            }
        }
	}
}

tester::~tester()
{
	for (auto& thread : threads_)
	{
		thread.join();
	}
}

tester::tester(
			size_t workers
		,   std::ofstream& log
	)
		:   dataset_(log)
		,   stack_ {}
		,   params_ {}
		,   threads_{}
		,   rem_(dataset_.size())
		,   mse_sum_(0)
		,   cce_sum_(0)
		,   acc_sum_(0)
{
	std::ifstream file(NNFILE, std::ios::binary);

    using clock = std::chrono::steady_clock;

	for (const auto& sample : dataset_)
	{
		stack_.emplace(sample);
	}
 
	if (!file) 
	{
		log
			<< "Unable to open neural network file!" 
			<< std::endl;

		std::exit(EXIT_FAILURE);
	}

	read(file, params_.data, PARAM_COUNT);

	if (file.eof())
	{
		log
			<< "Unable to read parameters from file!" 
			<< std::endl;

		std::exit(EXIT_FAILURE);
	}

	for (size_t i=0; i<workers; i++)
	{
		threads_.emplace_back(&tester::iterator, this);
	}

	auto start = clock::now();

    std::unique_lock<std::mutex> lock(mtx_);
	cv_.wait(lock, [this]() { return rem_==0; });

	auto end = clock::now();

	auto mse_error = mse_sum_ / dataset_.size();
	auto cce_error = cce_sum_ / dataset_.size();
	auto accuracy = acc_sum_ / dataset_.size();

	auto seconds = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();

	struct { int hr, min, sec; } elapsed;

	elapsed.hr = seconds / 3600;
	elapsed.min = (seconds % 3600) / 60;
	elapsed.sec = seconds % 60;

	log
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
}

int main(int argc, char *argv[])
{
	size_t workers=4;

	std::ofstream log(LOGFILE, std::ios::app);

	if (!log) exit(EXIT_FAILURE);

	if (argc >= 3) 
	{
		std::string param = argv[1];
		std::string value = argv[2];

		if (param=="workers")
		{
			try {
				workers = std::stoull(value);
			}
			catch (const std::invalid_argument& e) {

				log << value << " is not a number\n";
			}
			catch (const std::out_of_range& e) {
                
				log << value << " is too large\n";
			}

            if (!workers) {
                log << "There are not workers to evaluate the model." << std::endl;
                std::exit(EXIT_FAILURE);
            }
		}
	}

    tester(workers, log);
}