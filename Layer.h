#pragma once
#include "ExtendedMatrix.h"
#include "ActivationFunction.h"

namespace fnn {
	using alg::Matrix; using alg::ExMatrix;
	using actf::ActfType;
	class Layer {
	public:
		Layer() : num_neurons(0), actf(ActfType::TSigmoid) {}
		Layer(const int& num_neurons, const ActfType& actf) : num_neurons(num_neurons), actf(actf) {};

		// accessor
		const ExMatrix& getWeight() const { return w; }
		const ExMatrix& getBias() const { return b; }
		ActfType getActf() const { return actf; }
		int getNNeurons() const { return num_neurons; }

		// mutator
		void rebuild(const int& num_neurons, const int& fan_in);
		void setActf(const ActfType& actf) { Layer::actf = actf; }

		// weight initialize
		void Xavier_Initailize(const R& fan_in, const R& fan_out);
		void He_Initialize(const R& fan_in);

	protected:
		int num_neurons;
		ActfType actf;
		  
		ExMatrix w, b; // weight, bias
		ExMatrix f, grad;
	};
}