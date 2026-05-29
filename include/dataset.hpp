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
#include "policy.hpp"
#include "neural.hpp"

class sample
{

private:

	const board brd_;
	const move best_move_;
	const value eval_;

public:

	board_descriptor input();
	neural_output output();

	sample(std::ifstream& file);

};

struct input_data 
{
	const sample& sample;
	float * const grad;

public:

	input_data(const sample&, float[]);

};

using batch=std::queue<input_data>;

using samples=std::vector<const sample*>;

class dataset : public samples
{

private:

	std::list<sample> data_;

public:

	dataset(const char directory[]);

};

#endif // DATASET_HPP