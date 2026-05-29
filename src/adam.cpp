/*
 * Copyright (c) 2026 Ugwu Valentine Nnannaya.
 * All rights reserved.
 *
 * This project is proprietary. Unauthorized copying,
 * modification, or distribution is not permitted.
 */

#include <string>
#include <cstring>
#include <Eigen/Dense>

#include <io.hpp>
#include <adam.hpp>


bool adam::save() const
{
	std::ofstream file(ADAM_FILE, std::ios::binary);

	if (!file) {
		return false;
	}

	write(file, momentum_, PARAM_COUNT);
	write(file, velocity_, PARAM_COUNT);
	write(file, &discounted_gamma_, 1);
	write(file, &discounted_beta_, 1);

	return true;
}

bool adam::load()
{
	std::ifstream file(ADAM_FILE, std::ios::binary);

	if (!file) {
		return false;
	}

	read(file, momentum_, PARAM_COUNT);
	read(file, velocity_, PARAM_COUNT);
	read(file, &discounted_gamma_, 1);
	read(file, &discounted_beta_, 1);

	return !file.eof();
}

void adam::reset()
{
	std::memset((void*)momentum_, 0x0, sizeof(float) * PARAM_COUNT);
	std::memset((void*)velocity_, 0x0, sizeof(float) * PARAM_COUNT);
}

void adam::zero_grad()
{
	std::memset((void*)grad_, 0x0, sizeof(float) * PARAM_COUNT);
}

void adam::step()
{
	Eigen::Map<Eigen::VectorXf> momentum(momentum_, PARAM_COUNT);
	Eigen::Map<Eigen::VectorXf> velocity(velocity_, PARAM_COUNT);
	Eigen::Map<Eigen::VectorXf> temp1(temp1_, PARAM_COUNT);
	Eigen::Map<Eigen::VectorXf> temp2(temp2_, PARAM_COUNT);
	Eigen::Map<Eigen::VectorXf> params(params_, PARAM_COUNT);
	Eigen::Map<Eigen::VectorXf> grad(grad_, PARAM_COUNT);

	temp1.noalias() = beta_ * momentum_;
	temp2.noalias() = (1 - beta_) * grad_;

	momentum.noalias() = temp1 + temp2;

	temp1.noalias() = gamma_ * velocity_;
	temp2.array() = grad.array().square();
	temp2.array() *= 1 - gamma_;

	velocity.noalias() = temp1 + temp2;

	temp1.noalias() = momentum / (1 - discounted_beta_);
	temp2.noalias() = velocity / (1 - discounted_gamma_);

	temp1.array() *= alpha_;
	temp2.array() = temp2.array().sqrt();
	temp2.array() += epsilon_;

	params.array() -= temp1.array() / temp2.array();

	discounted_gamma_ *= gamma_;
	discounted_beta_ *= beta_;
}

adam::adam(
			float params[]
		,   float grad[]
		,   float alpha
		,   float beta
		,   float gamma
		,   float epsilon
	)
		:   params_(params)
		,   grad_(grad)
		,   momentum_ {}
		,   velocity_ {}
		,   temp1_ {}
		,   temp2_ {}
		,   discounted_beta_(beta)
		,   discounted_gamma_(gamma)
		,   alpha_(alpha)
		,   beta_(beta)
		,   gamma_(gamma)
		,   epsilon_(epsilon)
{}