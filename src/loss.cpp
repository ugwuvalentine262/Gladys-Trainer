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

    static float forward(const WDL& y_hat, const WDL& y)
    {
        auto max = std::max(y_hat.win, std::max(y_hat.draw, y_hat.loss));
        return -(y.win*y_hat.win + y.draw*y_hat.draw + y.loss*y_hat.loss) 
            +  max + std::log(std::exp(y_hat.win-max)+std::exp(y_hat.draw-max)+std::exp(y_hat.loss-max));
    }

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

	static WDL backward(const WDL& y_hat, const WDL& y)
    {
        auto max = std::max(y_hat.win, std::max(y_hat.draw, y_hat.loss));

        auto kw = std::exp(y_hat.win - max);
        auto kd = std::exp(y_hat.win - max);
        auto kl = std::exp(y_hat.win - max);

        auto sum = kw + kd + kl;

        return WDL(
                    kw / sum - y.win
                ,   kd / sum - y.draw
                ,   kl / sum - y.loss
            );
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

struct mae
{
	static float forward(const WDL& y_hat, const WDL& y)
	{
		return std::fabs(y_hat.q() - y.q());
	}
};

error loss::forward(const neural_output& y_hat, const neural_output& y)
{
	return  error(
					mae::forward(y_hat.wdl, y.wdl)
				,   cce::forward(y_hat.wdl, y.wdl)
				,   cce::forward(y_hat.z, y.z)
				,   y_hat.z.accuracy(y.z, 1)
				,   y_hat.z.accuracy(y.z, 3)
			);
}

neural_output loss::backward(const neural_output& y_hat, const neural_output& y)
{
	return  neural_output(
					cce::backward(y_hat.wdl, y.wdl)
				,   cce::backward(y_hat.z, y.z)
			);
}
