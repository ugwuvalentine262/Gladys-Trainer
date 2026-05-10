#ifndef LEARNER_NEURAL_HPP
#define LEARNER_NEURAL_HPP

#include <logicnn.h>
#include <logicnn_backprop.h>

#include "board_descriptor.hpp"

namespace chokochess::learner {

using policy_map=std::array<float, 14>;
using evaluation=float;
using gradients = std::vector<float>;
using parameters = std::vector<float>;

class forward_pass 
{
    friend class backward_pass;

private: 

	nn_msg_pass_t mp_[4];

	const parameters& params_;

public:

	forward_pass(const parameters& prams);
};

class backward_pass
{

private:
	const forward_pass& fp_;

    gradients& grad_;

public:

	backward_pass(const forward_pass& fp, gradients& grad);
};

} // namespace chokochess::learner

#endif // LEARNER_NEURAL_HPP