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

class adam
{

private:

	parameters& params_;
	gradients& grad_;

	std::vector<float> velocity_;
	std::vector<float> momentum_;

	float gamma_pow_;
	float beta_pow_;

	const float gamma_;
	const float beta_;
	const float alpha_;
	const float eps_;

    const size_t N_;

public:

	bool save(const std::string& filename);

	bool load(const std::string& filename);

	void step();

	adam(
			size_t N
		,   parameters& params
		,   gradients& grad
		,   float alpha
		,   float beta 
		,   float gamma 
		,   float epsilon
	);

};

#endif // ADAM_HPP