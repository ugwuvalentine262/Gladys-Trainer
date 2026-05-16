/*
 * Copyright (c) 2026 Ugwu Valentine Nnannaya.
 * All rights reserved.
 *
 * This project is proprietary. Unauthorized copying,
 * modification, or distribution is not permitted.
 */

#include <sstream>
#include <iomanip>
#include <ctime>

#include "summary.hpp"

std::string summary::format_info() const
{
	std::ostringstream oss;

	std::tm* t = std::localtime(&time_stamp);

	char timestamp[100];

	struct { int hr, min, sec } elapsed;

	elapsed.hr = seconds / 3600;
	elapsed.min = (seconds % 3600) / 60;
	elapsed.sec = seconds % 60;

	std::strftime(timestamp, sizeof(timestamp), "%a %b %d %H:%M:%S %Y", t);

	oss << std::fixed 
		<< std::setprecision(6)
		<< " | " 
		<< timestamp
		<< " | "
		<< std::setw(12)
		<< mse_loss
		<< "    | ";

	if (cce_loss>=0) 
	{
		oss << std::setw(12) << cce_loss << "     | ";
	}
	else
	{
		oss << "        -        | ";
	}

	if (accuracy>=0) 
	{
		oss << std::setw(12) << accuracy << "    | ";
	}
	else
	{
		oss << "       -        | ";
	}

	oss << std::setw(11)
		<< no_of_samples
		<< "    | "
		<< std::setfill('0')
		<< std::setw(2)
		<< elapsed.hr 
		<< ":"
		<< std::setw(2)
		<< elapsed.min
		<< ":"
		<< std::setw(2)
		<< elapsed.sec;
		<< std::endl;

	oss.str();
}

summary::summary
	(
			uint64_t samples
		,   uint64_t secs
		,   float    mse_err
		,   float    cce_err
		,   float    acc
	)
		:   time_stamp(std::time(nullptr))
		,   no_of_samples(samples)
		,   seconds(secs)
		,   mse_loss(mse_err)
		,   cce_loss(cce_err)
		,   accuracy(acc)
{}

summary::summary
	(
			uint64_t samples
		,   uint64_t secs
		,   float    mse_err
	)
		:   summary
	(
			samples
		,   sec
		,   mse_err
		,   -1
		,   -1
	)
{}