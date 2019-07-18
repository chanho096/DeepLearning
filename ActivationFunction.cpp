#include "ActivationFunction.h"
#include <cmath>

namespace actf {
	R ReLU(const R& x) {
		return MAX((R)0, x);
	}

	R dReLU(const R& x) {
		if (x > (R)0) return x;
		return (R)0;
	}

	R Sigmoid(const R& x) {
		return (R)1 / ((R)1 + exp(-1 * x));
	}

	R dSigmoid(const R& y) {
		return ((R)1 - y) * y;
	}

}