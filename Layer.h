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
		const ExMatrix& getActiveValue() const { return f; }
		ActfType getActf() const { return actf; }
		int getNNeurons() const { return num_neurons; }
		
		// mutator
		void rebuild(const int& num_neurons, const int& fan_in);
		void setActf(const ActfType& actf) { Layer::actf = actf; }

		// weight initialize
		void Xavier_Initialize(const R& fan_in, const R& fan_out);
		void He_Initialize(const R& fan_in);

		// main function
		void Set_Input(const ExMatrix& input); // use for input-layer
		void Set_Gradient(const ExMatrix& label, const Layer* const prev); // use for output-layer
		void Forward_Propagation(const Layer* const prev);
		void Backward_Propagation(const Layer* const prev, const Layer* const next);
		void Weight_Update(const R learning_rate);

	protected:
		int num_neurons;
		ActfType actf;
		  
		ExMatrix w, b; // weight, bias
		ExMatrix z_tmp, f;
		ExMatrix grad_z, grad_w, grad_b; // the derivative of cost

		void activate();
		void getGradientOfActf();
		void getGradientOfParameter(const Layer* const prev);
		void softmax();
	};
}