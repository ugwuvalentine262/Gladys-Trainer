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

#include "board_descriptor.hpp"
#include "policy.hpp"

#define EMBEDDING_DIM 32
#define LAYERS_COUNT 3

#define MATRIX_SIZE (EMBEDDING_DIM*EMBEDDING_DIM)
#define LAYER_PARAM_SIZE (MATRIX_SIZE * 2 * EDGE_COUNT)
#define WDL_OUTPUT_DIM 3
#define EMBEDDING_PARAM_IDX 0
#define ENCODER_PARAM_IDX (EMBEDDING_DIM*ATTRIBUTE_COUNT)
#define WDL_PARAM_IDX (ENCODER_PARAM_IDX + LAYER_PARAM_SIZE * LAYERS_COUNT)
#define WDL_CLASSIFIER_PARAM_IDX (WDL_PARAM_IDX + LAYER_PARAM_SIZE)
#define POLICY_PARAM_IDX (WDL_CLASSIFIER_PARAM_IDX + EMBEDDING_DIM * 2 * WDL_OUTPUT_DIM)
#define POLICY_CLASSIFIER_PARAM_IDX (WDL_CLASSIFIER_PARAM_IDX + LAYER_PARAM_SIZE)
#define PARAM_COUNT (POLICY_CLASSIFIER_PARAM_IDX + 0)
#define ATTENTION_OFFSET (MATRIX_SIZE * EDGE_COUNT)

using value=float;
using gradients = alignas(32) float[PARAM_COUNT];
using parameters = alignas(32) float[PARAM_COUNT];

struct neural_output
{
	const logits z;
	const value v;

public:

	neural_output(const value v, const logits& z);

};

class forward_pass
{
	friend class backward_pass;

private: 

	nn_embedding_t emb_;

	nn_msg_pass_t mp1_, mp2_, mp3_;
	nn_msg_pass_t policy_;

	const float *wdl_classifier_;
	const float *policy_classifier_;

	std::vector<float> wdl_temp_;
	std::vector<float> policy_temp_;
	std::vector<float> mp1_temp_;
	std::vector<float> mp2_temp_;
	std::vector<float> mp3_temp_;

public:

	neural_output operator()(const board_descriptor& brd);

	forward_pass(const float prams[]);

};

class backward_pass
{

private:

	const forward_pass& fp_;

	nn_msg_pass_backprop_t mp_;

	nn_msg_pass_backprop_t wdl_;
	nn_msg_pass_backprop_t value_;

	std::vector<float> temp_;

public:

	void operator()(const neural_output& dEdy, float grad[]);

	backward_pass(const forward_pass& fp);

};

#endif // NEURAL_HPP