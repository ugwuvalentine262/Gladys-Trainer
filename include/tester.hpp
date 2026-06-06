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
#include <atomic>
#include <string>
#include <stack>

#include "dataset.hpp"
#include "neural.hpp"

class tester 
{

private:

	std::ofstream& log_;
	dataset dataset_;
	std::stack<const sample*> stack_;
	parameters params_;
	std::list<std::thread> threads_;
	std::condition_variable cv_;
	std::mutex mtx_;
	std::atomic<int> it_;
	std::atomic<float> mse_sum_;
	std::atomic<float> cce_sum_;
	std::atomic<float> acc_sum_;

private:

	void iterator();

public:

	void test();

	~tester();

	tester(size_t workers, std::ofstream& log);

};

#endif // TESTER_HPP