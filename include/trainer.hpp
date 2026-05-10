#ifndef LEARNER_TRAINER_HPP
#define LEARNER_TRAINER_HPP

#include <string>
#include <chrono>
#include <cstdint>
#include <iostream>

#include "adam.hpp"
#include "propagator.hpp"
#include "dataset.hpp"
#include "neural.hpp"

namespace chokochess::learner {

class trainer
{
	using propagators=std::vector<propagator>;

public:

	struct epoch_result 
	{
		const uint64_t no_of_samples;
		const uint64_t seconds;
		const float cce_loss;
		const float mse_loss;
		const float accuracy;

		const std::string format_info;

		static constexpr char headers[] =
			"---------------------------------------------------------------------------------------------------------\n"
			"     Time-stamp     | Value-Loss(MSE) | Policy-Loss(CCE) | Policy-Accuracy | No. of samples | Time taken \n"
			"---------------------------------------------------------------------------------------------------------\n";
 
	public:

		epoch_result(uint64_t _seconds, uint64_t _minutes);

	};

private:

	parameters& params_;
	propagators props_;
	training_samples& dataset_;

private:

	void shuffle();

public:

	void one_epoch();

	trainer(
			size_t N
		,   training_samples& dataset
		,   parameters& params
    );

};

} // namespace chokochess::learner

#endif // LEARNER_TRAINER_HPP