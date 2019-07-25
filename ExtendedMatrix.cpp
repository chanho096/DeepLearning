#include "ExtendedMatrix.h"
#include "ActivationFunction.h"
#include <cstdlib>
#include <cmath>

namespace alg {
	std::random_device ExMatrix::rd;
	ExMatrix::engine ExMatrix::generator(rd());

	void ExMatrix::randomize() {
		// uniform distribution
		dis_uniform dis((R)0.0, (R)1.0);
		for (int i = 0; i < size; ++i) {
			values[i] = dis(generator);
		}
	}

	void ExMatrix::randomize(const R& mean, const R& stddev) {
		// normal distribution
		dis_normal dis(mean, stddev);
		for (int i = 0; i < size; ++i) {
			values[i] = dis(generator);
		}
	}

	void ExMatrix::normalize(const R& min, const int& axis) {

	}

	R ExMatrix::average() const {
		R avg = R(0.0);
		for (int i = 0; i < size; ++i) avg += values[i];
		return avg / size;
	}

	R ExMatrix::variance() const {
		R var = R(0.0); R avg = average();
		for (int i = 0; i < size; ++i) var += values[i] * values[i];
		return (var / size) - (avg * avg);
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

	void ExMatrix::log() {
		for (int i = 0; i < size; ++i) {
			values[i] = (R)std::log(values[i]);
		}
	}

	void ExMatrix::exp() {
		for (int i = 0; i < size; ++i) {
			values[i] = (R)std::exp(values[i]);
		}
	}
}
