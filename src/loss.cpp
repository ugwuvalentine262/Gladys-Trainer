/*
 * Copyright (c) 2026 Ugwu Valentine Nnannaya.
 * All rights reserved.
 *
 * This project is proprietary. Unauthorized copying,
 * modification, or distribution is not permitted.
 */

#include <cstring>
#include <Eigen/Dense>

#include <loss.hpp>
#include <policy.hpp>

class cce
{

public:

	static float forward(const logits& z, const policy& y)
	{
		Eigen::Matrix<float, POLICY_DIM, 1> z_, y_;

		std::memcpy(z_.data(), z.data(), sizeof(float) * POLICY_DIM);
		std::memcpy(y_.data(), y.data(), sizeof(float) * POLICY_DIM);

		auto max=z_.maxCoeff();

		y_ = z_.array() - max;
		y_ = y_.array().exp();

		auto h = max + std::log(y_.sum());

		z_.array() -= h;

		return -z_.dot(y_);
	}

	static logits backward(const logits& z, const policy& y)
	{
		auto h = softmax(z);

		Eigen::Map<Eigen::VectorXf> u(h.data(), h.size());
 		Eigen::Map<Eigen::VectorXf> v(y.data(), y.size());

 		u -= v;

		return h; 
	}

};

class mse
{

public:

	static float forward(float y_hat, float y)
	{
		float x = y_hat - y;
		return x * x;
	}

	static float backward(float y_hat, float y)
	{
		return (y_hat - y) * 2.0f;
	}

};

error loss::forward(const neural_output& y_hat, const neural_output& y)
{
	return  error(
					mse::forward(y_hat.v, y.v)
				,   cce::forward(y_hat.z, y.z)
				,   accurate(y_hat.z, y.z) ? 1 : 0
			);
}

neural_output loss::backward(const neural_output& y_hat, const neural_output& y)
{
	return  neural_output(
					mse::backward(y_hat.v, y.v)
				,   cce::backward(y_hat.z, y.z)
			);
}
