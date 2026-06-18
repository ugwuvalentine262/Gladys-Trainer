/*
 * Copyright (c) 2026 Ugwu Valentine Nnannaya.
 * All rights reserved.
 *
 * This project is proprietary. Unauthorized copying,
 * modification, or distribution is not permitted.
 */

#ifndef NEURAL_HPP
#define NEURAL_HPP

#include <vector>
#include <array>

#include <logicnn.h>
#include <logicnn_backprop.h>

#include "descriptor.hpp"
#include "policy.hpp"

#define EMBEDDING_DIM 16

#if ((EMBEDDING_DIM <= 0) || (EMBEDDING_DIM % 4))
    #error "Embedding dimension must be a multiple of 4"
#endif

#define MATRIX_SIZE (EMBEDDING_DIM*EMBEDDING_DIM)
#define LAYER_PARAM_SIZE (MATRIX_SIZE * 2 * EDGE_COUNT)
#define WDL_OUTPUT_DIM 3
#define EMBEDDING_PARAM_OFFSET 0
#define LAYER1_PARAM_OFFSET (EMBEDDING_DIM*ATTRIBUTE_COUNT)
#define LAYER2_PARAM_OFFSET (LAYER1_PARAM_OFFSET + LAYER_PARAM_SIZE)
#define LAYER3a_PARAM_OFFSET (LAYER2_PARAM_OFFSET + LAYER_PARAM_SIZE)
#define LAYER3b_PARAM_OFFSET (LAYER3a_PARAM_OFFSET + LAYER_PARAM_SIZE)
#define WDL_PARAM_OFFSET (LAYER3b_PARAM_OFFSET + LAYER_PARAM_SIZE)
#define WDL_BIAS_OFFSET (WDL_PARAM_OFFSET + EMBEDDING_DIM * 2 * WDL_OUTPUT_DIM)
#define POLICY_PARAM_OFFSET (WDL_BIAS_OFFSET + 3)
#define PARAM_COUNT (POLICY_PARAM_OFFSET) //(POLICY_PARAM_OFFSET + MATRIX_SIZE * 14)
#define ATTENTION_OFFSET (MATRIX_SIZE * EDGE_COUNT)

using value=float;

struct alignas(16) my_array {
    float data[PARAM_COUNT];
};

using gradients=my_array;
using parameters=my_array;

struct neural_output
{
	const value v;
	const logits z;

public:

	neural_output(
            const value& _v
        ,   const logits& _z
    );

};

class forward_pass
{
	friend class backward_pass;

private: 

	nn_embedding_t emb_;

	nn_msg_pass_t mp1_;
    nn_msg_pass_t mp2_;
    nn_msg_pass_t mp3a_;
    nn_msg_pass_t mp3b_;

	const nn_float_t *wdl_;
    const nn_float_t *wdl_b_;
    const nn_float_t *policy_;

    alignas(16) nn_float_t X1[NODE_COUNT][EMBEDDING_DIM];
    alignas(16) nn_float_t X2[NODE_COUNT][EMBEDDING_DIM];
    alignas(16) nn_float_t X3a[NODE_COUNT][EMBEDDING_DIM];
    alignas(16) nn_float_t X3b[NODE_COUNT][EMBEDDING_DIM];
    alignas(16) nn_float_t Xwdl[EMBEDDING_DIM * 2];
                nn_float_t y_wdl[3];

    int fcont_[EMBEDDING_DIM];
    int econt_[EMBEDDING_DIM];

    const descriptor *d_;

	std::vector<nn_byte_t> mp1_temp_;
	std::vector<nn_byte_t> mp2_temp_;
	std::vector<nn_byte_t> mp3a_temp_;
	std::vector<nn_byte_t> mp3b_temp_;

public:

	neural_output operator()(const descriptor& brd);

	forward_pass(const float params[]);

};

class backward_pass
{

private:

	const forward_pass& fpass_;

	nn_msg_pass_backprop_t mp_;

	std::vector<nn_byte_t> temp_;

public:

	void operator()(const neural_output& dEdy, float grad[]);

	backward_pass(const forward_pass& fpass);

};

#endif // NEURAL_HPP