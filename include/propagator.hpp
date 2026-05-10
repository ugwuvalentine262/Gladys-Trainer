#ifndef LEARNER_PROPAGATOR_HPP
#define LEARNER_PROPAGATOR_HPP

#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>

#include <logicnn.h>
#include <logicnn_backprop.h>

#include "cce.hpp"
#include "mse.hpp"
#include "dataset.hpp"
#include "neural.hpp"

namespace chokochess::learner {

class propagator
{
    
private:

    const parameters& params_;

    gradients grad_;

	std::thread thread_;
	std::mutex& mtx_;
	std::condition_variable& cv_;
	training_samples& samples_;
	bool& exit_;

	forward_pass fpass_;
	backward_pass bpass_;

private:

    void idle_loop();

public:

    void accumulate(gradients& grad) const;

    propagator(
            std::mutex& mtx 
        ,   std::condition_variable& cv 
        ,   training_samples& samples 
        ,   bool& exit
    );

};

} // namespace chokochess::learner

#endif // LEARNER_PROPAGATOR_HPP