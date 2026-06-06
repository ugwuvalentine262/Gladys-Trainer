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
#include <descriptor.hpp>

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

#define MESSAGE_BACKWARD(_a) \
    nn_msg_pass_backprop ( \
            &mp_ \
        ,   &fpass_.mp##_a##_ \
        ,   &fpass_.d_->graphnet \
        ,   fpass_.X##_a[0] \
        ,   dY[0] \
        ,   dX[0] \
        ,   grad + LAYER##_a##_PARAM_OFFSET \
        ,   grad + LAYER##_a##_PARAM_OFFSET + ATTENTION_OFFSET \
        ,   fpass_.mp##_a##_temp_.data() \
        ,   temp_.data() \
    ); \
    std::memcpy(dY, dX, sizeof(dX)) 

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
    Eigen::Map<const Eigen::MatrixXf> W_(W, out, in);
    Eigen::Map<const Eigen::MatrixXf> x_(x, in, batch);
    Eigen::Map<      Eigen::MatrixXf> y_(y, out, batch);

    y_.noalias() = W_ * x_;
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
    Eigen::Map<const Eigen::MatrixXf> a(A, common, Acols);
    Eigen::Map<const Eigen::MatrixXf> b(B, common, Bcols);
    Eigen::Map<      Eigen::MatrixXf> y(Y, Acols, Bcols);

    y.noalias() = a.transpose() * b;
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
    Eigen::Map<const Eigen::MatrixXf> a(A, Arows, common);
    Eigen::Map<const Eigen::MatrixXf> b(B, Brows, common);
    Eigen::Map<      Eigen::MatrixXf> y(Y, Arows, Brows);

    y.noalias() = a * b.transpose();
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
    Eigen::Map<const Eigen::RowVectorXf> v_(v, in);
    Eigen::Map<const Eigen::MatrixXf> W_(W, in, out);
    Eigen::Map<      Eigen::RowVectorXf> y_(y, out);

    y_.noalias() = v_ * W_;
}

static void softmax_backprop(
            float           dEdxy[]
        ,   float     const y[]
        ,   unsigned  const dim
    )
{
    Eigen::Map<const Eigen::VectorXf> y_(y, dim);
    Eigen::Map<      Eigen::VectorXf> dxy(dEdxy, dim);

    auto dot= y_.dot(dxy);

    dxy.array() -= dot;
    dxy *= y_;
}

void softmax
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

neural_output::neural_output
    (
            const value& _v
        ,   const logits& _z
    )
        :   v(_v)
        ,   z(_z)
{}

neural_output forward_pass::operator()(const descriptor& d)
{
    d_=&d;

    logits p;

    alignas(16) float Y[NODE_COUNT][EMBEDDING_DIM];

    Eigen::Map<
            const Eigen::Matrix<float, 3, EMBEDDING_DIM * 2, Eigen::RowMajor>
        > WDL(wdl_);

    Eigen::Map<const Eigen::Matrix<float, EMBEDDING_DIM * 2, 1>> x(Xwdl);
    Eigen::Map<Eigen::Vector3f> y(y_wdl);

    Eigen::Map<Eigen::VectorXf> friends(Xwdl, EMBEDDING_DIM);
    Eigen::Map<Eigen::VectorXf> enemies(Xwdl+EMBEDDING_DIM, EMBEDDING_DIM);

    auto& win=y[0];
    auto& draw=y[1];
    auto& loss=y[2];

    nn_embedding(
            &emb_
        ,   d.attributes
        ,   Y[0]
        ,   NODE_COUNT
    );

    auto required = nn_required_bytes(&mp1_, &d.graphnet);

    MESSAGE_PASSING(1);
    MESSAGE_PASSING(2);
    MESSAGE_PASSING(3);

    mp4a_temp_.resize(required);
    //mp4b_temp_.resize(required);
    std::memcpy(X4a[0], Y, sizeof(Y));
    //std::memcpy(X4b[0], Y, sizeof(Y));
    nn_msg_pass(&mp4a_, &d.graphnet, X4a[0], Y[0], mp4a_temp_.data());

    friends.setConstant(-9E9);
    enemies.setConstant(-9e9);

    for (int s=0; s< 64; s++)
    {
        if (d.b.mailbox[s]!=0)
        {
            if (d.b.mailbox[s] & 0x8) 
            {
                for (int i = 0; i < EMBEDDING_DIM; i++)
                {
                    if (Y[s][i] > enemies[i])
                    {
                        enemies[i] = Y[s][i];
                        econt_[i] = s;
                    }
                }
            }
            else {
                for (int i = 0; i < EMBEDDING_DIM; i++)
                {
                    if (Y[s][i] > friends[i])
                    {
                        friends[i] = Y[s][i];
                        fcont_[i] = s;
                    }
                }
            }
        }
    }

    //nn_msg_pass(&mp4b_, &d.graphnet, X4b[0], Y[0], mp4b_temp_.data());
    y.noalias() = WDL * x;

    win += wdl_b_[0];
    draw += wdl_b_[1];
    loss += wdl_b_[2];

    auto max = std::max(win, std::max(draw, loss));

    win = std::exp(win - max);
    draw = std::exp(draw - max);
    loss = std::exp(loss - max);

    auto sum = win + draw + loss;

    win /= sum;
    draw /= sum;
    loss /= sum;

    return neural_output(win - loss, p);
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
    ,   Xwdl {}
    ,   fcont_ {}
    ,   econt_ {}
    ,   d_(0)
    ,   mp1_temp_ {}
    ,   mp2_temp_ {}
    ,   mp3_temp_ {}
    ,   mp4a_temp_ {}
    ,   mp4b_temp_ {}
{}

void backward_pass::operator()(const neural_output& dEdy, float grad[])
{
    alignas(16) float dY[NODE_COUNT][EMBEDDING_DIM]={};
    alignas(16) float dX[NODE_COUNT][EMBEDDING_DIM]={};
    alignas(16) float dy[3] = {dEdy.v, 0, -dEdy.v};
    alignas(16) float dy_mp[EMBEDDING_DIM * 2];

    float *friends = dy_mp;
    float *enemies = dy_mp + EMBEDDING_DIM;

    auto dot = dy[0] * fpass_.y_wdl[0]
            +  dy[2] * fpass_.y_wdl[2];

    dy[0] = fpass_.y_wdl[0] * (dy[0] - dot);
    dy[1] = fpass_.y_wdl[1] * (dy[1] - dot);
    dy[2] = fpass_.y_wdl[2] * (dy[2] - dot);

    grad[WDL_BIAS_OFFSET+0] = dy[0];
    grad[WDL_BIAS_OFFSET+1] = dy[1];
    grad[WDL_BIAS_OFFSET+2] = dy[2];

    Eigen::Map<const Eigen::Vector3f> dy_(dy);

    Eigen::Map<
            const Eigen::Matrix<float, 1, EMBEDDING_DIM * 2>
        > x(fpass_.Xwdl);

    Eigen::Map<
            Eigen::Matrix<float, 3, EMBEDDING_DIM * 2, Eigen::RowMajor>
        > dWDL(grad + WDL_PARAM_OFFSET);

    dWDL.noalias() = dy_ * x;

    matmat_mult
    (
            fpass_.wdl_
        ,   dy
        ,   dy_mp
        ,   1
        ,   3
        ,   2 * EMBEDDING_DIM
    );

    for (int i = 0; i < EMBEDDING_DIM; i++)
    {
        dY[fpass_.fcont_[i]][i] = friends[i];
        dY[fpass_.econt_[i]][i] = enemies[i];
    }

    temp_.resize(fpass_.mp1_temp_.size());

    MESSAGE_BACKWARD(4a);
    MESSAGE_BACKWARD(3 );
    MESSAGE_BACKWARD(2 );
    MESSAGE_BACKWARD(1 );

    nn_embedding_backprop(
            &fpass_.emb_
        ,   fpass_.d_->attributes
        ,   dY[0]
        ,   grad + EMBEDDING_PARAM_OFFSET
        ,   NODE_COUNT
    );
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
