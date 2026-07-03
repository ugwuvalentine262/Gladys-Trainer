/*
 * Copyright (c) 2026 Ugwu Valentine Nnannaya.
 * All rights reserved.
 *
 * This project is proprietary. Unauthorized copying,
 * modification, or distribution is not permitted.
 */

#include <Eigen/Dense>
#include <string>

#include "neural.hpp"
#include "descriptor.hpp"

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

    dxy.array() *= y_.array();
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
    Eigen::Map<      Eigen::VectorXf> v(data, count);
    Eigen::Map<const Eigen::VectorXf> h(delta, count);

    v += h;
}

void mlp::backward(const float dEdy[], float dEdx[], float grad[], float tmp[], size_t batch) const
{
    std::memset(grad, 0, sizeof(float)*param_count());

    auto dEW1=grad;
    auto dEb1=dEW1+i_dim_*h_dim_;
    auto dEW2=dEb1+h_dim_;
    auto dEb2=dEW2+h_dim_*o_dim_;

    for (size_t i=0; i < batch; i++)
    {
        accumulate(dEb2, dEdy + i * o_dim_, o_dim_);
    }

    matmatT_mult(dEdy, x2, dEW2, o_dim_, h_dim_, batch);
    matTmat_mult(W2_, dEdy, tmp, h_dim_, batch, o_dim_);

    Eigen::Map<const Eigen::ArrayXf> v(x2, h_dim_*batch);
    Eigen::Map<      Eigen::ArrayXf> delta(tmp, h_dim_*batch);

    delta = (v > 0.0f).select(delta, 0.01f * delta);

    for (size_t i=0; i < batch; i++)
    {
        accumulate(dEb1, tmp + i * h_dim_, h_dim_);
    }

    matmatT_mult(tmp, x1, dEW1, h_dim_, i_dim_, batch);
    matTmat_mult(W1_, tmp, dEdx, i_dim_, batch, h_dim_);
}

void mlp::forward(const float x[], float y[], size_t batch)
{
    std::memcpy(x1, x, sizeof(float) * batch * i_dim_);
    matmat_mult(W1_, x1, x2, batch, i_dim_, h_dim_);

    for (size_t i=0; i < batch; i++) {
        accumulate(x2 + i * h_dim_, b1_, h_dim_);
    }

    Eigen::Map<Eigen::ArrayXf> v(x2, h_dim_*batch);

    v = (v > 0.0f).select(v, 0.01 * v);

    matmat_mult(W2_, x2, y, batch, h_dim_, o_dim_);

    for (size_t i=0; i < batch; i++) {
        accumulate(y + i * o_dim_, b2_, o_dim_);
    }
}

size_t mlp::param_count() const 
{
    return i_dim_*h_dim_+h_dim_*o_dim_+h_dim_+o_dim_;
}

mlp::mlp
    (
            const float params[]
        ,   size_t i_dim
        ,   size_t h_dim
        ,   size_t o_dim
        ,   float meta[]
        ,   size_t max_batch
    )
    :   W1_(params)
    ,   b1_(W1_+i_dim*h_dim)
    ,   W2_(b1_+h_dim)
    ,   b2_(W2_+h_dim*o_dim)
    ,   i_dim_(i_dim)
    ,   h_dim_(h_dim)
    ,   o_dim_(o_dim)
    ,   x1(meta)
    ,   x2(meta+i_dim*max_batch)
{}

void value_head::backward(float dEdy, float dEdX[][EMBEDDING_DIM], float grad[]) const
{
    float dy[3] = {dEdy, 0, -dEdy};
    float dx[2][EMBEDDING_DIM] = {};
    float tmp[WDL_INPUT_DIM*WDL_HIDDEN_DIM];

    auto dot = dy[0] * y_wdl_[0]
            +  dy[2] * y_wdl_[2];

    dy[0] = y_wdl_[0] * (dy[0] - dot);
    dy[1] = y_wdl_[1] * (dy[1] - dot);
    dy[2] = y_wdl_[2] * (dy[2] - dot);

    wdl_.backward(dy, dx[0], grad, tmp, 1);

    std::memcpy(dEdX[G1], dx[0], sizeof(float) * EMBEDDING_DIM);
    std::memcpy(dEdX[G2], dx[1], sizeof(float) * EMBEDDING_DIM);
}

float value_head::forward(float X[][EMBEDDING_DIM])
{
    nn_float_t x[2][EMBEDDING_DIM];

    auto& win=y_wdl_[0];
    auto& draw=y_wdl_[1];
    auto& loss=y_wdl_[2];

    std::memcpy(x[0], X[G1], sizeof(float) * EMBEDDING_DIM);
    std::memcpy(x[1], X[G2], sizeof(float) * EMBEDDING_DIM);

    wdl_.forward(x[0], y_wdl_, 1);

    auto max = std::max(win, std::max(draw, loss));

    win = std::exp(win - max);
    draw = std::exp(draw - max);
    loss = std::exp(loss - max);

    auto sum = win + draw + loss;

    win /= sum;
    draw /= sum;
    loss /= sum;

    return win - loss;
}

value_head::value_head(const float params[])
    : wdl_(params, WDL_INPUT_DIM, WDL_HIDDEN_DIM, WDL_OUTPUT_DIM, meta_, 1)
{}

void policy_head::backward(const descriptor& d, const logits& dEdy, float dEdX[][EMBEDDING_DIM], float grad[]) const
{
    float dy[128];
    float dyp[22*PROMO_OUTPUT_DIM];
    float dEdx[128][POLICY_INPUT_DIM];
    float dEdxp[22][PROMO_INPUT_DIM];
    float tmp[POLICY_INPUT_DIM*128];

    std::memcpy(dy, dEdy.vals_data(), sizeof(float) * d.mcount_);

    softmax_backprop(dy, y_, d.mcount_);
    int n=0;

    for (int i=0; i < d.mcount_; i++)
    {
        auto promo = d.moves_[i].promo;

        if (promo>0x1&&promo<0x6) {
            dyp[n++]=dy[i];
        }
    }

    if (n) {
        promotion_.backward(dyp, dEdxp[0], grad + policy_.param_count(), tmp, n/4);
    }
    policy_.backward(dy, dEdx[0], grad, tmp, d.mcount_);

    for (int i=0, k=0; i < d.mcount_; i++)
    {
        const auto& m = d.moves_[i];

        if (m.promo>0x1&&m.promo<0x6)
        {
            auto dx1=dEdxp[k];
            auto dx2=dx1+EMBEDDING_DIM;
            auto dx3=dx2+EMBEDDING_DIM;

            accumulate(dEdX[G3]    , dx1, EMBEDDING_DIM);
            accumulate(dEdX[m.from], dx2, EMBEDDING_DIM);
            accumulate(dEdX[m.to]  , dx3, EMBEDDING_DIM);

            for (int j=0; j<4; j++)
            {
                auto dx1=dEdx[i+j];
                auto dx2=dx1+EMBEDDING_DIM;
                auto dx3=dx2+EMBEDDING_DIM;

                accumulate(dEdX[G3]    , dx1, EMBEDDING_DIM);
                accumulate(dEdX[m.from], dx2, EMBEDDING_DIM);
                accumulate(dEdX[m.to]  , dx3, EMBEDDING_DIM);

                (void)j;
            }

            i += 3, k++;
        }
        else {
            auto dx1=dEdx[i];
            auto dx2=dx1+EMBEDDING_DIM;
            auto dx3=dx2+EMBEDDING_DIM;

            accumulate(dEdX[G3]    , dx1, EMBEDDING_DIM);
            accumulate(dEdX[m.from], dx2, EMBEDDING_DIM);
            accumulate(dEdX[m.to]  , dx3, EMBEDDING_DIM);
        }
    }
}

logits policy_head::forward(const descriptor& d, float X[][EMBEDDING_DIM])
{
    float x[128][POLICY_INPUT_DIM]={};
    float xp[22][PROMO_INPUT_DIM]={};
    float yp[22][PROMO_OUTPUT_DIM]={};

    int n=0;

    struct { float *y, *yp; } pr[22][PROMO_OUTPUT_DIM];

    for (int i=0; i < d.mcount_; i++)
    {
        const move& m = d.moves_[i];

        if (m.promo>0x1&&m.promo<0x6)
        {
            for (int k=0; k < 4; k++)
            {
                auto x1=x[i+k];
                auto x2=x1+EMBEDDING_DIM;
                auto x3=x2+EMBEDDING_DIM;

                std::memcpy(x1, X[G3], sizeof(float)*EMBEDDING_DIM);
                std::memcpy(x2, X[m.from], sizeof(float)*EMBEDDING_DIM);
                std::memcpy(x3, X[m.to], sizeof(float)*EMBEDDING_DIM);

                pr[n][k].y=y_+i+k;
                pr[n][k].yp=yp[n]+k;
            }
            std::memcpy(xp[n], x[i], sizeof(float)*POLICY_INPUT_DIM);
            i+=3, n++;
        }
        else {
            auto x1=x[i];
            auto x2=x1+EMBEDDING_DIM;
            auto x3=x2+EMBEDDING_DIM;

            std::memcpy(x1, X[G3], sizeof(float)*EMBEDDING_DIM);
            std::memcpy(x2, X[m.from], sizeof(float)*EMBEDDING_DIM);
            std::memcpy(x3, X[m.to], sizeof(float)*EMBEDDING_DIM);
        }
    }

    policy_.forward(x[0], y_, d.mcount_);

    if (n!=0) {
        promotion_.forward(xp[0], yp[0], n);

        for (int i=0; i < n; i++)
            for (int j=0; j < 4; j++)
                *pr[i][j].y += *pr[i][j].yp;
    }

    softmax(y_, d.mcount_);

    return logits(d.moves_, y_, d.mcount_);
}

policy_head::policy_head(const float params[])
    : policy_(params, POLICY_INPUT_DIM, POLICY_HIDDEN_DIM, 1, policy_meta_, 128)
    , promotion_(params + policy_.param_count(), PROMO_INPUT_DIM, PROMO_HIDDEN_DIM, PROMO_OUTPUT_DIM, promo_meta_, 22)
{}

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

    float Y[NODE_COUNT][EMBEDDING_DIM];

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

    return neural_output( 
                value_.forward(Y)
              , policy_.forward(d, Y)
           );
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
    ,   value_(params+WDL_PARAM_OFFSET)
    ,   policy_(params+POLICY_PARAM_OFFSET)
    ,   X1 {}
    ,   X2 {}
    ,   X3 {}
    ,   d_(0)
    ,   mp1_temp_ {}
    ,   mp2_temp_ {}
    ,   mp3_temp_ {}
{}

void backward_pass::operator()(const neural_output& dEdy, float grad[])
{
    float dY[NODE_COUNT][EMBEDDING_DIM]={};
    float dX[NODE_COUNT][EMBEDDING_DIM]={};

    fpass_.policy_.backward(*fpass_.d_, dEdy.z, dY, grad + POLICY_PARAM_OFFSET);
    fpass_.value_.backward(dEdy.v, dY, grad + WDL_PARAM_OFFSET);

    temp_.resize(fpass_.mp1_temp_.size());

    MESSAGE_BACKWARD(3);
    MESSAGE_BACKWARD(2);
    MESSAGE_BACKWARD(1);

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
