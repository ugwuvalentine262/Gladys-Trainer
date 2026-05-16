/*
 * Copyright (c) 2026 Ugwu Valentine Nnannaya.
 * All rights reserved.
 *
 * This project is proprietary. Unauthorized copying,
 * modification, or distribution is not permitted.
 */

#ifndef ADAM_HPP
#define ADAM_HPP

#include <string>
#include <vector>

#include "neural.hpp"

class adam
{

private:

	float * const params_;
	float * const grad_;

    float         data_[PARAM_COUNT*4];

	float         discounted_beta_;
	float         discounted_gamma_;

	float   const alpha_;
	float   const beta_;
	float   const gamma_;
	float   const epsilon_;

public:

	bool save() const;
	bool load();
	void step();
    void zero_grad();

	adam(
			float       * params
		,   float const * grad
		,   float const   alpha
		,   float const   beta 
		,   float const   gamma 
		,   float const   epsilon
	);

};

#endif // ADAM_HPP