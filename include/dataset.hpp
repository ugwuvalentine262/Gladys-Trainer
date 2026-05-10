/*
 * Copyright (c) 2026 Ugwu Valentine Nnannaya.
 * All rights reserved.
 *
 * This project is proprietary. Unauthorized copying,
 * modification, or distribution is not permitted.
 */

#ifndef DATASET_HPP
#define DATASET_HPP

#include <vector>
#include <list>
#include <fstream>
#include <array>

#include "board_descriptor.hpp"
#include "neural.hpp"

class sample
{

private:

	const board brd_;
	const move best_move_;
	const float eval_;

public:

	board_descriptor encoder_input();
	policy_map policy_target();
	evaluation value_target();

	sample(std::ifstream& ifs);
};

using training_samples=std::vector<const sample*>;

class dataset : public training_samples
{

private:

	std::list<sample> data_;

public:

	dataset(const std::string& filename);

};

#endif // DATASET_HPP