/*
 * Copyright (c) 2026 Ugwu Valentine Nnannaya.
 * All rights reserved.
 *
 * This project is proprietary. Unauthorized copying,
 * modification, or distribution is not permitted.
 */

#include <fstream>
#include <iomanip>
#include <ctime>
#include <string>

#include <dataset.hpp>
#include <tester.hpp>
#include <io.hpp>
#include <loss.hpp>

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
				if (it_>=dataset_.size())
				{
					cv_.notify_one();
				}
				break;
			}
			sample = stack_.top();
			stack_.pop();
		}

		auto x = sample->input();
		auto y = sample->output();

		auto y_hat = fpass(x);

		auto error = loss::forward(y_hat, y);

		mse_sum_ += error.mse;
		cce_sum_ += error.cce;
		acc_sum_ += error.accuracy;

		it_++;
	}
}

void tester::test()
{
	auto start = std::time(nullptr);

	{
		std::unique_lock<std::mutex> lock(mtx_);

		cv_.wait(lock, [this]() {
			return it_.load() >= dataset_.size() && stack_.empty();
		});
	}

	auto end = std::time(nullptr);

	auto mse_error = mse_sum_.load() / dataset_.size();
	//auto cce_error = cce_sum_.load() / dataset_.size();
	//auto accuracy = acc_sum_.load() / dataset_.size();

	size_t seconds = std::difftime(end, start);

	struct { int hr, min, sec; } elapsed;

	elapsed.hr = seconds / 3600;
	elapsed.min = (seconds % 3600) / 60;
	elapsed.sec = seconds % 60;

	log_ 
		<< std::fixed 
		<< std::setprecision(6)
		<< " mse: " 
		<< std::setw(8)
		<< mse_error
		/*<< " cce: "
		<< std::setw(8)
		<< cce_error
		<< " accuracy: "
		<< std::setw(8) 
		<< accuracy*/
		<< " elapsed: "
		<< std::setfill('0')
		<< std::setw(2)
		<< elapsed.hr 
		<< ":"
		<< std::setw(2)
		<< elapsed.min
		<< ":"
		<< std::setw(2)
		<< elapsed.sec;
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
		:   log_(log)
		,   dataset_(DATASET_DIR, log)
		,   stack_ {}
		,   params_ {}
		,   threads_{}
		,   it_ {0}
		,   mse_sum_ {0}
		,   cce_sum_ {0}
		,   acc_sum_ {0}
{
	std::ifstream file(NNFILE, std::ios::binary);

	for (const auto& sample : dataset_)
	{
		stack_.emplace(sample);
	}
 
	if (!file) 
	{
		log_ 
			<< "Unable to open neural network file!" 
			<< std::endl;

		exit(EXIT_FAILURE);
	}

	read(file, params_.data, PARAM_COUNT);

	if (file.eof())
	{
		log_ 
			<< "Unable to read parameters from file!" 
			<< std::endl;

		exit(EXIT_FAILURE);
	}

	for (size_t i=0; i<workers; i++)
	{
		threads_.emplace_back(tester::iterator, this);
	}
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
		}
	}

	tester tester(workers, log);

	tester.test();
}