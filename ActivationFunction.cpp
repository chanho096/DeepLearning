#include "ActivationFunction.h"
#include <math.h>

namespace actf {
	R ReLU(const R& x) {
		return MAX((R)0, x);
	}

	R dReLU(const R& x) {
		return (x > (R)0) ? x : (R)0;
	}

	R Sigmoid(const R& x) {
		return (R)1 / ((R)1 + exp(-1 * x));
	}

	R dSigmoid(const R& y) {
		return ((R)1 - y) * y;
	}
}