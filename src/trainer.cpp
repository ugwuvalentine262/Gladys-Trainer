/*
 * Copyright (c) 2026 Ugwu Valentine Nnannaya.
 * All rights reserved.
 *
 * This project is proprietary. Unauthorized copying,
 * modification, or distribution is not permitted.
 */

#include <fstream>
#include <iostream>
#include <chrono>
#include <Eigen/Dense>

#include <trainer.hpp>

summary trainer::train()
{
    
}

trainer::trainer()
{
    
}

int main(int argc, char *argv[])
{
	trainer trainer;

    std::ofstream ofs(LOG_FILE);

	for (;;) {

		summary summary = trainer.train();
	}
}