#include "Layer.h"
#include <cstdlib>
#include <ctime>
#include <math.h>

namespace fnn {
	void Layer::rebuild(const int& num_neurons, const int& fan_in) {
		Layer::num_neurons = num_neurons;
		w.initialize(num_neurons, fan_in);
		b.initialize(num_neurons, 1);
	}

	void Layer::Xavier_Initailize(const R& fan_in, const R& fan_out) {
		R tmp = sqrt((R)6.0 / (fan_in + fan_out));
		w.randomize(tmp * 2, tmp * -1);
	}

	void Layer::He_Initialize(const R& fan_in, const R& fan_out) {
		R tmp = sqrt((R)6.0 / fan_in);
		w.randomize(tmp * 2, tmp * -1);
	}

}