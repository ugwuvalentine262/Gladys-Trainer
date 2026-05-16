/*
 * Copyright (c) 2026 Ugwu Valentine Nnannaya.
 * All rights reserved.
 *
 * This project is proprietary. Unauthorized copying,
 * modification, or distribution is not permitted.
 */

#include <iostream>
#include <fstream>
#include <iomanip>

#include <tester.hpp>

summary tester::test()
{
    
}

tester::tester()
{
    
}

int main(int argc, char *argv[])
{
	tester tester;

	summary summary = tester.test();

    std::ofstream ofs(LOG_FILE);
}