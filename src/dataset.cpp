/*
 * Copyright (c) 2026 Ugwu Valentine Nnannaya.
 * All rights reserved.
 *
 * This project is proprietary. Unauthorized copying,
 * modification, or distribution is not permitted.
 */

#include <cmath>
#include <dataset.hpp>

descriptor sample::input() const
{
    return brd_;
}

neural_output sample::output() const
{
    return neural_output(eval_, best_move_);
}

sample::sample(
            const std::string& fen
        ,   const std::string& eval
        ,   const std::string& policy
    )
        :   brd_(fen)
        ,   best_move_(policy)
        ,   eval_(1.0 / (1 + std::exp(std::stoi(eval) * -0.004)))
{}

input_data::input_data(const sample& sample, float grad[])
    :   sample_(sample)
    ,   grad_(grad)
{}

dataset::dataset(const std::string directory, std::ofstream& log)
    :   data_ {}
{
    std::ifstream fen(directory + "/fen.txt");
    std::ifstream eval(directory + "/eval.txt");
    std::ifstream policy(directory + "/policy.txt");

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

        data_.emplace_back(fen_str, eval_str, policy_str);
        samples::emplace_back(&data_.back());
    }
}
