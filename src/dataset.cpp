/*
 * Copyright (c) 2026 Ugwu Valentine Nnannaya.
 * All rights reserved.
 *
 * This project is proprietary. Unauthorized copying,
 * modification, or distribution is not permitted.
 */

#include <random>
#include <algorithm>
#include <cmath>

#include <dataset.hpp>

descriptor sample::input() const
{
    return brd_;
}

neural_output sample::output() const
{
    return neural_output(eval_, policy_);
}

sample::sample(
            const std::string& fen
        ,   const std::string& eval
        ,   const std::string& policy
    )
        :   brd_(fen)
        ,   policy_(policy)
        ,   eval_(2.0 / (1 + std::exp(std::stoi(eval) * -0.004)) - 1)
{}

input_data::input_data(const sample& sample, float grad[])
    :   sample_(sample)
    ,   grad_(grad)
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

dataset::dataset(std::ofstream& log)
    :   data_ {}
{
    std::ifstream fen(DATASET_DIR "/fen.txt");
    std::ifstream eval(DATASET_DIR "/eval.txt");
    std::ifstream policy(DATASET_DIR "/policy.txt");

    if (!fen || !eval || !policy)
    {
        log << "Unable to open dataset" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string fen_str;
    std::string eval_str;
    std::string policy_str;

    while (std::getline(fen, fen_str))
    {
        if (!std::getline(policy, policy_str)) {
            log << "policy is inconsistent with fen" << std::endl;
            exit(EXIT_FAILURE);
        }

        if (!std::getline(eval, eval_str)) {
            log << "eval is inconsistent with fen" << std::endl;
            exit(EXIT_FAILURE);
        }

        try {
            data_.emplace_back(fen_str, eval_str, policy_str); 
        }
        catch(const policy::bad&)
        {
            log << "bad policy in dataset" << std::endl;
            exit(EXIT_FAILURE);
        }
        catch(const move::bad&)
        {
            log << "Illegal move in policy" << std::endl;
            exit(EXIT_FAILURE);
        }
        catch(...)
        {
            log << "Unkown error occurred while initializing dataset" << std::endl;
            exit(EXIT_FAILURE);
        }

        samples::emplace_back(&data_.back());
    }
}
