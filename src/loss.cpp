/*
 * Copyright (c) 2026 Ugwu Valentine Nnannaya.
 * All rights reserved.
 *
 * This project is proprietary. Unauthorized copying,
 * modification, or distribution is not permitted.
 */

#include <Eigen/Dense>

#include <loss.hpp>
#include <policy.hpp>

class cce
{

public:

	static float forward(const logits& z, const policy& y)
	{
		float error=0;

		Eigen::Map<const Eigen::VectorXf> z_(z.data(), z.size());
		Eigen::Map<const Eigen::VectorXf> y_(y.data(), y.size());

		auto max=z_.maxCoeff();

		y_ = z_.array() - max;
		y_ = y_.array().exp();

		auto h = max + std::log(y_.sum()); // log-sum-exponent

		for (auto k=0; k < POLICY_DIMp; k++) 
		{
			error += -y[k] * (z[k] - h);
		}

		return error;
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

value loss::forward(const neural_output& y_hat, const neural_output& y)
{
	
}

neural_output loss::backward(const neural_output& y_hat, const neural_output& y)
{
	
}
