/*
 * Copyright (c) 2026 Ugwu Valentine Nnannaya.
 * All rights reserved.
 *
 * This project is proprietary. Unauthorized copying,
 * modification, or distribution is not permitted.
 */

#include <Eigen/Dense>
#include <string>

#include <neural.hpp>
#include <board_descriptor.hpp>

#define INIT_MESSAGE_PASSER(_a) \
    mp##_a##_ { \
                .msg_computer=nn_msg_computer \
            ,   .msg_aggregator=nn_msg_aggregator \
            ,   .accumulate=accumulate \
            ,   .matmat_mult=matmat_mult \
            ,   .vecmat_mult=vecmat_mult \
            ,   .softmax=softmax \
            ,   .weights=params + LAYER##_a##_PARAM_OFFSET \
            ,   .attention=params + LAYER##_a##_PARAM_OFFSET + ATTENTION_OFFSET \
            ,   .dim=EMBEDDING_DIM \
            ,   .edge_count=EDGE_COUNT \
        }

#define MESSAGE_PASSING(_a) \
    mp##_a##_temp_.resize(required); \
    std::memcpy(X##_a[0], Y, sizeof(Y)); \
    nn_msg_pass(&mp##_a##_, &d.graphnet, X##_a[0], Y[0], mp##_a##_temp_.data());

static void matmat_mult
    (
            float    const * const W
        ,   float    const * const x
        ,   float          * const y
        ,   unsigned         const batch
        ,   unsigned         const in
        ,   unsigned         const out
    )
{
    
}

static void matTmat_mult
    (
            float    const * const A
        ,   float    const * const B
        ,   float          * const Y
        ,   unsigned         const Acols
        ,   unsigned         const Bcols
        ,   unsigned         const common
    )
{
    
}

static void matmatT_mult
    (
            float    const * const A
        ,   float    const * const B
        ,   float          * const Y
        ,   unsigned         const Arows
        ,   unsigned         const Brows
        ,   unsigned         const common
    )
{
    
}

static void vecmat_mult
    (
            float    const         v[]
        ,   float    const * const W
        ,   float                  y[]
        ,   unsigned const         in
        ,   unsigned const         out
    )
{
    
}

static void softmax_backprop(
            float           dEdxy[]
        ,   float     const y[]
        ,   unsigned  const dim
    )
{

}

static void softmax
    (
            float          xy[]
        ,   unsigned const dim
    )
{
	Eigen::Map<Eigen::VectorXf> y(xy, dim);

	auto max=y.maxCoeff();

	y = y.array() - max;
	y = y.array().exp();

	y /= y.sum();
}

static void accumulate
    (
            float          data[]
        ,   float    const delta[]
        ,   unsigned       count
    )
{
    Eigen::Map<Eigen::VectorXf> v(data, count);
    Eigen::Map<const Eigen::VectorXf> h(delta, count);

    v += h;
}

neural_output::neural_output(const value _v, const logits& _z)
    :   v(_v)
    ,   z(_z)
{}

neural_output::neural_output(const value _v, const move&)
    :   v(_v)
    ,   z{}
{}

neural_output::neural_output(const value _v)
    :   v(_v)
    ,   z{}
{}

neural_output forward_pass::operator()(const board_descriptor& d)
{
    brd_=&d;

    alignas(16) float Y[NODE_COUNT][EMBEDDING_DIM];
    alignas(16) float y_wdl[3];

    Eigen::Map<Eigen::VectorXf> friends(Xwdl, EMBEDDING_DIM);
    Eigen::Map<Eigen::VectorXf> enemies(Xwdl+EMBEDDING_DIM, EMBEDDING_DIM);

    float& win=y_wdl[0];
    float& draw=y_wdl[1];
    float& loss=y_wdl[2];

    nn_embedding(
            &emb_
        ,   d.attributes
        ,   Y[0]
        ,   NODE_COUNT
    );

    auto required = nn_required_bytes(&mp1_, &d.graphnet);

    MESSAGE_PASSING(1 );
    MESSAGE_PASSING(2 );
    MESSAGE_PASSING(3 );
    MESSAGE_PASSING(4a);

    friends.setConstant(-9e10);
    enemies.setConstant(-9e10);

    for (int i=0; i< 64; i++)
    {
        std::memcpy(k[i], Y[i], sizeof(float) * EMBEDDING_DIM);

        if (d.b.mailbox[i]!=0)
        {
            Eigen::Map<Eigen::VectorXf> h(k[i], EMBEDDING_DIM);

            if (d.b.mailbox[i] & 0x8) {
                enemies = enemies.cwiseMax(h);
            }
            else {
                friends = friends.cwiseMax(h);   
            }
        }
    }

    nn_matmat_mult(
            wdl_
        ,   Xwdl
        ,   y_wdl
        ,   1
        ,   EMBEDDING_DIM
        ,   3
    );

    auto max = std::max(win, std::max(draw, loss));

    win = std::exp(win - max);
    draw = std::exp(draw - max);
    loss = std::exp(loss - max);

    auto sum = win + draw + loss;

    win /= sum;
    draw /= sum;
    loss /= sum;

    return neural_output(win - loss);
}

forward_pass::forward_pass(const float params[])
    :   emb_ {
                .accumulate=accumulate
            ,   .data=params + EMBEDDING_PARAM_OFFSET
            ,   .attribute_count=ATTRIBUTE_COUNT
            ,   .dim=EMBEDDING_DIM
        }
    ,   INIT_MESSAGE_PASSER(1)
    ,   INIT_MESSAGE_PASSER(2)
    ,   INIT_MESSAGE_PASSER(3)
    ,   INIT_MESSAGE_PASSER(4a)
    ,   INIT_MESSAGE_PASSER(4b)
    ,   wdl_(params+WDL_PARAM_OFFSET)
    ,   wdl_b_(wdl_+ EMBEDDING_DIM * 2 * WDL_OUTPUT_DIM)
    ,   policy_(wdl_b_+3)
    ,   X1 {}
    ,   X2 {}
    ,   X3 {}
    ,   X4a {}
    ,   X4b {}
    ,   k {}
    ,   Xwdl {}
    ,   brd_(0)
    ,   mp1_temp_ {}
    ,   mp2_temp_ {}
    ,   mp3_temp_ {}
    ,   mp4a_temp_ {}
    ,   mp4b_temp_ {}
{}

void backward_pass::operator()(const neural_output& dEdy, float grad[])
{

}

backward_pass::backward_pass(const forward_pass& fpass)
    :   fpass_(fpass)
    ,   mp_ {
                .softmax=softmax_backprop
            ,   .matTmat_mult=matTmat_mult
            ,   .matmatT_mult=matmatT_mult
        }
    ,   temp_ {}
{}
