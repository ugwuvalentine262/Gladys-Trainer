/*
 * Copyright (c) 2026 Ugwu Valentine Nnannaya.
 * All rights reserved.
 *
 * This project is proprietary. Unauthorized copying,
 * modification, or distribution is not permitted.
 */

#include <cstring>
#include <Eigen/Dense>

#include "loss.hpp"
#include "policy.hpp"

class cce
{

public:

	static float forward(const logits& z, const logits& y)
	{
        alignas(16) float _z[128], _y[128];

        Eigen::Map<Eigen::VectorXf> z_(_z, z.size());
        Eigen::Map<Eigen::VectorXf> y_(_y, y.size());

		std::memcpy(_z, z.vals_data(), sizeof(float) * z.size());
		std::memcpy(_y, y.vals_data(), sizeof(float) * y.size());

		auto max=z_.maxCoeff();

		y_ = z_.array() - max;
		y_ = y_.array().exp();

		auto h = max + std::log(y_.sum());

		z_.array() -= h;

		return -z_.dot(y_);
	}

	static logits backward(const logits& z, const logits& y)
	{
		alignas(16) float h[128];

        std::memcpy(h, z.vals_data(), sizeof(float) * z.size());

	    Eigen::Map<Eigen::VectorXf> p(h, z.size());

	    auto max=p.maxCoeff();

	    p = p.array()-max;
	    p = p.array().exp();
	    p /= p.sum();

		Eigen::Map<Eigen::VectorXf> u(h, z.size());
 		Eigen::Map<const Eigen::VectorXf> v(y.vals_data(), y.size());

 		u -= v;

		return logits(z.moves_data(), h, z.size()); 
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
				,   y_hat.z == y.z ? 1 : 0
			);
}

neural_output loss::backward(const neural_output& y_hat, const neural_output& y)
{
	return  neural_output(
					mse::backward(y_hat.v, y.v)
				,   cce::backward(y_hat.z, y.z)
			);
}
