/*
 * Copyright (c) 2026 Ugwu Valentine Nnannaya.
 * All rights reserved.
 *
 * This project is proprietary. Unauthorized copying,
 * modification, or distribution is not permitted.
 */

#include <fstream>
#include <iostream>
#include <algorithm>
#include <random>

#include "descriptor.hpp"
#include "dataset.hpp"

descriptor sample::input() const
{
    policy pi(policy_, brd_.white);
    return descriptor(brd_, pi.moves_data(), pi.size());
}

neural_output sample::output() const
{
    policy pi(policy_, brd_.white);
    return neural_output(wdl_, pi);
}

sample::sample(
            const std::string& fen
        ,   const std::string& wdl
        ,   const std::string& policy
    )
        :   brd_(fen)
        ,   policy_(policy)
        ,   wdl_(wdl)
{}

input_data::input_data(const sample *sample, gradients& grad)
    :   sample_(sample)
    ,   grad_(&grad)
{}

void dataset::shuffle()
{
	std::random_device rd;
	std::mt19937 gen(rd());

    std::shuffle
    (
            samples::begin()
        ,   samples::end()
        ,   gen
    );
}

dataset::dataset()
    :   data_ {}
{
    std::ifstream fen(DATASET_DIR "/fen.txt");
    std::ifstream wdl(DATASET_DIR "/wdl.txt");
    std::ifstream policy(DATASET_DIR "/policy.txt");

    if (!fen || !wdl || !policy)
    {
        std::cerr << "Unable to open dataset" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string fen_str;
    std::string wdl_str;
    std::string policy_str;

    while (std::getline(fen, fen_str))
    {
        if (!std::getline(policy, policy_str)) {
            std::cerr << "policy is inconsistent with fen" << std::endl;
            std::exit(EXIT_FAILURE);
        }

        if (!std::getline(wdl, wdl_str)) {
            std::cerr << "wdl is inconsistent with fen" << std::endl;
            std::exit(EXIT_FAILURE);
        }

        try {
            data_.emplace_back(fen_str, wdl_str, policy_str); 
        }
        catch(...)
        {
            std::cerr << "Unkown error occurred while initializing dataset" << std::endl;
            std::exit(EXIT_FAILURE);
        }

        samples::emplace_back(&data_.back());
    }
}
