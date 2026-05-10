#ifndef LEARNER_TESTER_HPP
#define LEARNER_TESTER_HPP

#include <vector>
#include <thread>
#include <atomic>
#include <string>

#include "dataset.hpp"
#include "neural.hpp"

namespace chokochess::learner {

class tester 
{

public:

	struct result
	{
		const uint64_t seconds;
		const uint64_t no_of_samples;
		const float cce_loss;
		const float mse_loss;
		const float accuracy;

		const std::string format_info;

		static constexpr char headers[] =
			"-------------------------------------------------------------------------------------------------------\n"
			"    Time-stamp    | Value-Loss(MSE) | Policy-Loss(CCE) | Policy-Accuracy | No. of samples | Time taken \n"
			"-------------------------------------------------------------------------------------------------------\n";
 
	public:

		result(uint64_t seconds, uint64_t no_of_samples);
	};

private:

	std::vector<std::thread> threads_;
	std::atomic<int> idx_;
	std::atomic<float> err_sum_;
    
	const dataset& dataset_;
    parameters params_;

private:

    void run_test();

public:

	~tester();
	tester();

};

} // namespace chokochess::tester

#endif // LEARNER_TESTER_HPP