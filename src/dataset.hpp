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
#include <array>
#include <queue>
#include <string>

#include "WDL.hpp"
#include "descriptor.hpp"
#include "policy.hpp"
#include "neural.hpp"

class sample
{

	using Policy = std::string;

private:

	const board brd_;
	const Policy policy_;
	const WDL wdl_;

public:

	descriptor input() const;
	neural_output output() const;

	sample(
                const std::string& fen
            ,   const std::string& wdl
            ,   const std::string& policy
        );

};

struct input_data 
{
	const sample * sample_=0;
	gradients * grad_=0;

public:

    input_data()=default;
	input_data(const sample *, gradients&);

};

using batch=std::queue<input_data>;

using samples=std::vector<const sample*>;

class dataset : public samples
{

private:

	std::list<sample> data_;

public:

	void shuffle();

	dataset();

};

#endif // DATASET_HPP