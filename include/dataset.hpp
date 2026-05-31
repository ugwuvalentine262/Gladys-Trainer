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
#include <queue>
#include <string>

#include "descriptor.hpp"
#include "policy.hpp"
#include "neural.hpp"

class sample
{

private:

	const board brd_;
	const move best_move_;
	const value eval_;

public:

	descriptor input() const;
	neural_output output() const;

	sample(
                const std::string& fen
            ,   const std::string& eval
            ,   const std::string& policy
        );

};

struct input_data 
{
	const sample& sample_;
	float * const grad_;

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

	dataset(const std::string directory, std::ofstream& log);

};

#endif // DATASET_HPP