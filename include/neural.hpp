/*
 * Copyright (c) 2026 Ugwu Valentine Nnannaya.
 * All rights reserved.
 *
 * This project is proprietary. Unauthorized copying,
 * modification, or distribution is not permitted.
 */

#ifndef NEURAL_HPP
#define NEURAL_HPP

#include <logicnn.h>
#include <logicnn_backprop.h>

#include "board_descriptor.hpp"

using policy_map=std::array<float, 14>;
using evaluation=float;
using gradients = std::vector<float>;
using parameters = std::vector<float>;

class forward_pass 
{
    friend class backward_pass;

private: 

	nn_msg_pass_t mp_[4];

	const parameters& params_;

public:

	forward_pass(const parameters& prams);
};

class backward_pass
{

private:
	const forward_pass& fp_;

    gradients& grad_;

public:

	backward_pass(const forward_pass& fp, gradients& grad);
};

#endif // NEURAL_HPP