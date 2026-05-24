/*
 * Copyright (c) 2026 Ugwu Valentine Nnannaya.
 * All rights reserved.
 *
 * This project is proprietary. Unauthorized copying,
 * modification, or distribution is not permitted.
 */

#include <string>
#include <Eigen/Dense>
#include <fstream>
#include <cstring>
#include <adam.hpp>

static inline write(std::ofstream& file, const float data[], size_t N)
{
	file.write(reinterpret_cast<const char*>(data), sizeof(float) * N);
}

static inline read(std::ifstream& file, float data[], size_t N)
{
	file.read(reinterpret_cast<char*>(data), sizeof(float) * N);
}

static inline bool is_little_endian()
{
	uint16_t value = 0x0001U;
	return *reinterpret_cast<uint8_t*>(&value) == 1;
}

static inline float reverse_bytes(float f)
{
	uint32_t bits;

	std::memcpy(&bits, &f, sizeof(bits));

	bits =  ((bits & 0x000000FFu) << 24) |
			((bits & 0x0000FF00u) << 8)  |
			((bits & 0x00FF0000u) >> 8)  |
			((bits & 0xFF000000u) >> 24);

	std::memcpy(&f, &bits, sizeof(bits));

	return f;
}

bool adam::save() const
{
	std::ofstream file(ADAM_FILE, std::ios::binary);

	if (!file) {
		return false;
	}

	if (is_little_endian()) 
	{
 		write(file, data_, PARAM_COUNT * 2);
 		write(file, &discounted_gamma_, 1);
 		write(file, &discounted_beta_, 1);
	}
 	else
 	{
 		for (int i=0; i < PARAM_COUNT * 2; i++)
 		{
 			float f = reverse_bytes(data_[i]);
 			write(file, &f, 1);
 		}

 		float g = reverse_bytes(discounted_gamma_);
 		float b = reverse_bytes(discounted_beta_);

 		write(file, &g, 1);
 		write(file, &b, 1);
	}

	return true;
}

bool adam::load()
{
	std::ifstream file(ADAM_FILE, std::ios::binary);

	if (!file) {
		return false;
	}

    read(file, data_, PARAM_COUNT * 2);
 	read(file, &discounted_gamma_, 1);
 	read(file, &discounted_beta_, 1);

 	if (!is_little_endian())
	{
 		for (int i=0; i < PARAM_COUNT * 2; i++)
 		{
 			data_[i] = reverse_bytes(data_[i]);
 		}

 		discounted_gamma_ = reverse_bytes(discounted_gamma_);
 		discounted_beta_ = reverse_bytes(discounted_beta_);
	}

	return !file.eof();
}

void adam::zero_grad()
{
	std::memset((void*)grad_, 0x0, sizeof(float) * PARAM_COUNT);
}

void adam::step()
{
	Eigen::Map<Eigen::VectorXf> momentum(data_+PARAM_COUNT*0, PARAM_COUNT);
	Eigen::Map<Eigen::VectorXf> velocity(data_+PARAM_COUNT*1, PARAM_COUNT);
	Eigen::Map<Eigen::VectorXf> temp1(data_+PARAM_COUNT*2, PARAM_COUNT);
	Eigen::Map<Eigen::VectorXf> temp2(data_+PARAM_COUNT*3, PARAM_COUNT);
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
			float * params
		,   float * grad
		,   float   alpha
		,   float   beta
		,   float   gamma
		,   float   epsilon
	)   
		:   params_(params)
		,   grad_(grad)
		,   data_ {}
		,   discounted_beta_(beta)
		,   discounted_gamma_(gamma)
		,   alpha_(alpha)
		,   beta_(beta)
		,   gamma_(gamma)
		,   epsilon_(epsilon)
{}