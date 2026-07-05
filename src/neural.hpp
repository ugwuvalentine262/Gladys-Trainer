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

#include "WDL.hpp"
#include "descriptor.hpp"
#include "policy.hpp"

#if ((EMBEDDING_DIM <= 0) || (EMBEDDING_DIM % 8))
    #error "Embedding dimension must be a multiple of 8"
#endif

#define WDL_INPUT_DIM (EMBEDDING_DIM*2)
#define WDL_HIDDEN_DIM (EMBEDDING_DIM*2)
#define WDL_OUTPUT_DIM 3
#define POLICY_INPUT_DIM (EMBEDDING_DIM*3)
#define POLICY_HIDDEN_DIM (EMBEDDING_DIM*2)
#define PROMO_INPUT_DIM (EMBEDDING_DIM*3)
#define PROMO_HIDDEN_DIM (EMBEDDING_DIM)
#define PROMO_OUTPUT_DIM 4
#define MATRIX_SIZE (EMBEDDING_DIM*EMBEDDING_DIM)
#define LAYER_PARAM_SIZE (MATRIX_SIZE * 2 * EDGE_COUNT)
#define WDL_OUTPUT_DIM 3
#define EMBEDDING_PARAM_OFFSET 0
#define LAYER1_PARAM_OFFSET (EMBEDDING_DIM*ATTRIBUTE_COUNT)
#define LAYER2_PARAM_OFFSET (LAYER1_PARAM_OFFSET + LAYER_PARAM_SIZE)
#define LAYER3_PARAM_OFFSET (LAYER2_PARAM_OFFSET + LAYER_PARAM_SIZE)
#define WDL_PARAM_OFFSET (LAYER3_PARAM_OFFSET + LAYER_PARAM_SIZE)
#define POLICY_PARAM_OFFSET (WDL_PARAM_OFFSET + (WDL_INPUT_DIM * WDL_HIDDEN_DIM) + (WDL_HIDDEN_DIM * WDL_OUTPUT_DIM) + WDL_OUTPUT_DIM + WDL_HIDDEN_DIM)
#define PROMO_PARAM_OFFSET (POLICY_PARAM_OFFSET + (POLICY_INPUT_DIM * POLICY_HIDDEN_DIM) + (POLICY_HIDDEN_DIM) + 1 + POLICY_HIDDEN_DIM)
#define PARAM_COUNT (PROMO_PARAM_OFFSET + (PROMO_INPUT_DIM * PROMO_HIDDEN_DIM) + (PROMO_HIDDEN_DIM * PROMO_OUTPUT_DIM) + PROMO_HIDDEN_DIM + PROMO_OUTPUT_DIM)
#define ATTENTION_OFFSET (MATRIX_SIZE * EDGE_COUNT)

using value=float;

struct alignas(16) my_array {
    float data[PARAM_COUNT];
};

using gradients=my_array;
using parameters=my_array;

struct neural_output
{
	const WDL wdl;
	const logits z;

public:

	neural_output(
            const WDL& _wdl
        ,   const logits& _z
    );

};

class mlp
{
    const float *W1_;
    const float *b1_;
    const float *W2_;
    const float *b2_;

    const size_t i_dim_;
    const size_t h_dim_;
    const size_t o_dim_;

    float *x1, *x2;

public:

    void backward(const float dEdY[], float dEdX[], float grad[], float tmp[], size_t batch) const;
    void forward(const float x[], float y[], size_t batch);

    size_t param_count() const;

    mlp(
                const float params[]
            ,   size_t i_dim
            ,   size_t h_dim
            ,   size_t o_dim
            ,   float meta[]
            ,   size_t max_batch
        );
};

class wdl_head 
{
    mlp wdl_;

    nn_float_t meta_[WDL_INPUT_DIM+WDL_HIDDEN_DIM];

public:

    void backward(WDL dEdy, float dEdX[][EMBEDDING_DIM], float grad[]) const;
    WDL forward(float Y[][EMBEDDING_DIM]);

    wdl_head(const float params[]);
};

class policy_head
{
    mlp policy_, promotion_;

    nn_float_t y_[128];
    nn_float_t policy_meta_[POLICY_INPUT_DIM*128+POLICY_HIDDEN_DIM*128];
    nn_float_t promo_meta_[PROMO_INPUT_DIM*22+PROMO_HIDDEN_DIM*22];

public:

    void backward(const descriptor& d, const logits& dEdy, float dEdX[][EMBEDDING_DIM], float grad[]) const;
    logits forward(const descriptor& d, float X[][EMBEDDING_DIM]);

    policy_head(const float params[]);
};

class forward_pass
{
	friend class backward_pass;

private: 

	nn_embedding_t emb_;

	nn_msg_pass_t mp1_;
    nn_msg_pass_t mp2_;
    nn_msg_pass_t mp3_;

    wdl_head wdl_;
    policy_head policy_;

    nn_float_t X1[NODE_COUNT][EMBEDDING_DIM];
    nn_float_t X2[NODE_COUNT][EMBEDDING_DIM];
    nn_float_t X3[NODE_COUNT][EMBEDDING_DIM];

    const descriptor *d_;

	std::vector<nn_byte_t> mp1_temp_;
	std::vector<nn_byte_t> mp2_temp_;
	std::vector<nn_byte_t> mp3_temp_;

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