/*
 * Copyright (c) 2026 Ugwu Valentine Nnannaya.
 * All rights reserved.
 *
 * This project is proprietary. Unauthorized copying,
 * modification, or distribution is not permitted.
 */

#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include <string>
#include <list>
#include <stack>
#include <condition_variable>
#include <mutex>
#include <thread>

#include "dataset.hpp"
#include "io.hpp"
#include "loss.hpp"
#include "neural.hpp"

class tester 
{

private:

	dataset dataset_;
	std::stack<const sample*> stack_;
	parameters params_;
	std::list<std::thread> threads_;
	std::condition_variable cv_;
	std::mutex mtx_;
	size_t rem_;
	float q_mae_;
	float wdl_cce_;
	float pi_cce_;
	float pi_accuracy1_;
	float pi_accuracy3_;

private:

	void iterator();

public:

	~tester();

	tester(size_t workers);

};

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

            q_mae_ += error.q_mae;
            wdl_cce_ += error.wdl_cce;
            pi_cce_ += error.pi_cce;
            pi_accuracy1_ += error.pi_accuracy1;
            pi_accuracy3_ += error.pi_accuracy3;

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

tester::tester(size_t workers)
		:   dataset_()
		,   stack_ {}
		,   params_ {}
		,   threads_{}
		,   rem_(dataset_.size())
	    ,   q_mae_(0)
	    ,   wdl_cce_(0)
	    ,   pi_cce_(0)
	    ,   pi_accuracy1_(0)
	    ,   pi_accuracy3_(0)
{
	std::ifstream file(NNFILE, std::ios::binary);

    using clock = std::chrono::steady_clock;

	for (const auto& sample : dataset_)
	{
		stack_.emplace(sample);
	}
 
	if (!file) 
	{
		std::cerr << "Unable to open neural network file!" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	read(file, params_.data, PARAM_COUNT);

	if (file.eof())
	{
		std::cerr << "Unable to read parameters from file!" << std::endl;
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
		<< std::setprecision(7)
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
        << " | "
		<< std::setfill('0')
		<< std::setw(2)
		<< elapsed.hr 
		<< ":"
		<< std::setw(2)
		<< elapsed.min
		<< ":"
		<< std::setw(2)
		<< elapsed.sec
		<< std::setfill(' ')
        << std::endl;
}

int main(int argc, char *argv[])
{
	size_t workers=4;

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

				std::clog << value << " is not a number\n";
			}
			catch (const std::out_of_range& e) {
                
				std::clog << value << " is too large\n";
			}

            if (!workers) {
                std::cerr << "There are no workers to evaluate the model." << std::endl;
                std::exit(EXIT_FAILURE);
            }
		}
	}

    {
        tester tester(workers);
    }
}