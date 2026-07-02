/*
 * Copyright (c) 2026 Ugwu Valentine Nnannaya.
 * All rights reserved.
 *
 * This project is proprietary. Unauthorized copying,
 * modification, or distribution is not permitted.
 */

#include <cstring>
#include <Eigen/Dense>
#include <cmath>

#include "loss.hpp"
#include "policy.hpp"

class cce
{

public:

	static float forward(const logits& z, const logits& y)
	{
        float _z[128], _y[128], _x[128];

        Eigen::Map<const Eigen::VectorXf> u(_z, z.size());
        Eigen::Map<const Eigen::VectorXf> v(_y, y.size());
        Eigen::Map<      Eigen::VectorXf> w(_x, y.size());

		std::memcpy(_z, z.vals_data(), sizeof(float) * z.size());
		std::memcpy(_y, y.vals_data(), sizeof(float) * y.size());
    
		auto max=u.maxCoeff();

		w = u.array() - max;
		w = w.array().exp();

        return -v.dot(u) + max + std::log(w.sum());
	}

	static logits backward(const logits& z, const logits& y)
	{
		float h[128];

        std::memcpy(h, z.vals_data(), sizeof(float) * z.size());

 		Eigen::Map<const Eigen::VectorXf> v(y.vals_data(), y.size());
	    Eigen::Map<      Eigen::VectorXf> u(h, z.size());

	    auto max=u.maxCoeff();

	    u = u.array()-max;
	    u = u.array().exp();
	    u /= u.sum();
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

class mae
{

public:

	static float forward(float y_hat, float y)
	{
		return std::fabs(y_hat - y);
	}
};

error loss::forward(const neural_output& y_hat, const neural_output& y)
{
	return  error(
					mse::forward(y_hat.v, y.v)
				,   mae::forward(y_hat.v, y.v)
				,   cce::forward(y_hat.z, y.z)
				,   y_hat.z.accuracy(y.z, 1)
				,   y_hat.z.accuracy(y.z, 3)
			);
}

neural_output loss::backward(const neural_output& y_hat, const neural_output& y)
{
	return  neural_output(
					mse::backward(y_hat.v, y.v)
				,   cce::backward(y_hat.z, y.z)
			);
}
