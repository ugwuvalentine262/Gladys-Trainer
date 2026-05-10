#ifndef LEARNER_DATASET_HPP
#define LEARNER_DATASET_HPP

#include <vector>
#include <list>
#include <fstream>
#include <array>

#include <chess/board.hpp>

#include "board_descriptor.hpp"
#include "neural.hpp"

namespace chokochess::learner {

class sample
{

private:

	const chess::board brd_;
	const chess::move best_move_;
	const float eval_;

public:

	board_descriptor encoder_input();
	policy_map policy_target();
	evaluation value_target();

	sample(std::ifstream& ifs);
};

using training_samples=std::vector<const sample*>;

class dataset : public training_samples
{

private:

	std::list<sample> data_;

public:

	dataset(const std::string& filename);

};
    
} // namespace chokochess::learner

#endif // LEARNER_DATASET_HPP