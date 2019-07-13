#include "Layer.h"
#include <cstdlib>
#include <ctime>
#include <math.h>

namespace fnn {
	void Layer::rebuild(const int& num_neurons, const int& fan_in) {
		Layer::num_neurons = num_neurons;
		if (fan_in <= 0) return;
		w.initialize(num_neurons, fan_in);
		b.initialize(num_neurons, 1);
	}

	void Layer::Xavier_Initailize(const R& fan_in, const R& fan_out) {
		R tmp = sqrt((R)6.0 / (fan_in + fan_out));
		w.randomize(tmp * 2, tmp * -1);
	}

	void Layer::He_Initialize(const R& fan_in) {
		R tmp = sqrt((R)6.0 / fan_in);
		w.randomize(tmp * 2, tmp * -1);
	}

	void Layer::Set_Input(const ExMatrix &input) {
		assert(input.getNRows() == num_neurons);
		f.copy(input);
	}

	void Layer::Set_Gradient(const ExMatrix &lable, const Layer* const prev) {
		// set gradient of output layer.
		// Cross-Entropy loss function and Sigmoid (in output layer) are assumed.
		assert(lable.getNRows() == num_neurons);
		assert(lable.getNColumns() == f.getNColumns());
		Layer::grad_z.copy(lable);
		grad_z.multiply((R)-1);
		grad_z.addition(f);
		getGradientOfParameter(prev);
	}

	void Layer::Forward_Propagation(const Layer* const prev) {
		w.product(prev->f, z_tmp);
		z_tmp.addition(b);
		f.copy(z_tmp);
		activate();
	}

	void Layer::Backward_Propagation(const Layer* const prev, const Layer* const next) {
		next->w.productWithTranspose(next->grad_z, grad_z);
		getGradientOfActf();
		getGradientOfParameter(prev);
	}

	void Layer::Weight_Update(const R learning_rate) {
		R alpha = learning_rate * -1;
		grad_w.multiply(alpha);
		w.addition(grad_w);

		grad_b.multiply(alpha);
		b.addition(grad_b);
	}

	void Layer::activate() {
		switch (actf) {
		case ActfType::TSigmoid:
			f.Sigmoid(); break;
		case ActfType::TReLU:
			f.ReLU(); break;
		}
	}

	void Layer::getGradientOfActf() {
		switch (actf) {
		case ActfType::TSigmoid:
			z_tmp.dSigmoid(); break;
		case ActfType::TReLU:
			z_tmp.dReLU(); break;
		}
	}

	void Layer::getGradientOfParameter(const Layer* const prev) {
		R m = grad_z.getNColumns(); m = 1 / m;
		assert(m == prev->f.getNColumns());

		grad_z.productTransposed(prev->f, grad_w);
		grad_z.sum(grad_b, 1);

		grad_w.multiply(m);
		grad_b.multiply(m);
	}
}