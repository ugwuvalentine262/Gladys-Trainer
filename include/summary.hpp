#ifndef SUMMARY_HPP
#define SUMMARY_HPP

#include <cstdint>
#include <ctime>
#include <string>

struct summary 
{
	const std::time_t   time_stamp;
	const std::uint64_t no_of_samples;
	const std::uint64_t seconds;
	const std::uint64_t serial_no;
	const float         mse_loss;
	const float         cce_loss;
	const float         accuracy;

	static constexpr char headers[] =
		"---------------------------------------------------------------------------------------------------------------------\n"
		" S/N |        Time-stamp        | Value-Loss(MSE) | Policy-Loss(CCE) | Policy-Accuracy | No. of samples | Time taken \n"
		"---------------------------------------------------------------------------------------------------------------------\n";

private:

	std::string format_info() const;

public:

	summary(
			uint64_t samples
		,   uint64_t secs
		,   float    mse_err
		,   float    cce_err
		,   float    acc
	);

	summary(
			uint64_t samples
		,   uint64_t secs
		,   float    mse_err
	);
};

#endif // SUMMARY_HPP