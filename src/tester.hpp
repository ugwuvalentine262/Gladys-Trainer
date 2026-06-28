/*
 * Copyright (c) 2026 Ugwu Valentine Nnannaya.
 * All rights reserved.
 *
 * This project is proprietary. Unauthorized copying,
 * modification, or distribution is not permitted.
 */

#ifndef TESTER_HPP
#define TESTER_HPP

#include <fstream>
#include <chrono>
#include <list>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <string>
#include <stack>

#include "dataset.hpp"
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
	float mse_sum_;
	float mae_sum_;
	float cce_sum_;
	float acc_sum_;

private:

	void iterator();

public:

	~tester();

	tester(size_t workers, std::ofstream& log);

};

#endif // TESTER_HPP