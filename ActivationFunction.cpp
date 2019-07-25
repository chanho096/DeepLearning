#include "ActivationFunction.h"
#include <cmath>

namespace actf {
	R ReLU(const R& x) {
		return MAX((R)0.0, x);
	}

	R dReLU(const R& x) {
		if (x > (R)0.0) return (R)1.0;
		return (R)0.0;
	}

	R Sigmoid(const R& x) {
		return (R)1.0 / ((R)1.0 + exp((R)-1.0 * x));
	}

	R dSigmoid(const R& y) {
		return ((R)1.0 - y) * y;
	}

	R LReLU(const R& x) {
		if (x > (R)0.0) return x;
		return (R)0.01 * x;
	}

	R dLReLU(const R& x) {
		if (x > (R)0.0) return (R)1.0;
		return (R)0.01;
	}
}