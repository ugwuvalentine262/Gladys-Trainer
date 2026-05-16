/*
 * Copyright (c) 2026 Ugwu Valentine Nnannaya.
 * All rights reserved.
 *
 * This project is proprietary. Unauthorized copying,
 * modification, or distribution is not permitted.
 */

#ifndef TESTER_HPP
#define TESTER_HPP

#include <chrono>
#include <vector>
#include <thread>
#include <atomic>
#include <string>

#include "dataset.hpp"
#include "neural.hpp"

class tester 
{

private:

	std::vector<std::thread> threads_;
	std::atomic<int> idx_;
	std::atomic<float> mse_sum_;
	std::atomic<float> cce_sum_;
    std::atomic<float> acc_sum_;
    
	dataset dataset_;
	parameters params_;

public:

	summary test();

	tester();

};

#endif // TESTER_HPP