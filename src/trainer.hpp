/*
 * Copyright (c) 2026 Ugwu Valentine Nnannaya.
 * All rights reserved.
 *
 * This project is proprietary. Unauthorized copying,
 * modification, or distribution is not permitted.
 */

#ifndef TRAINER_HPP
#define TRAINER_HPP

#include <fstream>
#include <string>
#include <vector>
#include <list>

#include "adam.hpp"
#include "propagator.hpp"
#include "dataset.hpp"
#include "neural.hpp"
#include "adam.hpp"

class trainer
{
	using propagators=std::list<propagator>;

private:

	std::ofstream& log_;
	parameters params_;
	gradients grad_;
	std::vector<gradients> xgrad_;
	propagators props_;
	dataset dataset_;
	adam adam_;
	const size_t batch_size_;

	std::atomic<size_t> it_;
	std::atomic<float> mse_sum_;
	std::atomic<float> cce_sum_;
	std::atomic<float> acc_sum_;
	std::mutex mtx_;
	std::condition_variable cv_;
	batch batch_;
	bool stop_;

public:

	void train();

	~trainer();

	trainer(
            size_t workers
        ,   size_t batch_size
        ,   float alpha
        ,   std::ofstream& file
    );

};

#endif // TRAINER_HPP