#include "ExtendedMatrix.h"
#include "ActivationFunction.h"
#include <cstdlib>

namespace alg {
	void ExMatrix::randomize() {
		//srand((unsigned int)time(NULL));
		for (int i = 0; i < size; ++i) {
			values[i] = (R)std::rand() / (R)RAND_MAX;
		}
	}

	void ExMatrix::randomize(const R& scale, const R& min) {
		//srand((unsigned int)time(NULL));
		for (int i = 0; i < size; ++i) {
			values[i] = (R)std::rand() / (R)RAND_MAX * scale + min;
		}
	}

	void ExMatrix::normalize(const R& min, const int& axis) {

	}

	void ExMatrix::ReLU() {
		for (int i = 0; i < size; ++i) {
			values[i] = actf::ReLU(values[i]);
		}
	}

	void ExMatrix::dReLU() {
		for (int i = 0; i < size; ++i) {
			values[i] = actf::dReLU(values[i]);
		}
	}

	void ExMatrix::Sigmoid() {
		for (int i = 0; i < size; ++i) {
			values[i] = actf::Sigmoid(values[i]);
		}
	}

	void ExMatrix::dSigmoid() {
		for (int i = 0; i < size; ++i) {
			values[i] = actf::dSigmoid(values[i]);
		}
	}
}
