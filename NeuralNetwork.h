#pragma once
#include "Layer.h"

namespace fnn {
	class NeuralNetwork;
	class HyperParameter {
		friend class NeuralNetwork;
	public:
		HyperParameter() : num_neurons(nullptr) { initialize(); }
		HyperParameter(const int& num_layers) : num_neurons(nullptr) { initialize(num_layers); }
		HyperParameter(const HyperParameter& hp) : num_neurons(nullptr) { copy(hp); }
		~HyperParameter() { SAFE_DELETE_ARRAY(num_neurons); num_layers = 0; }
		
		void operator = (const HyperParameter& hp) { copy(hp); }
		void initialize();
		void initialize(const int& num_layers);
		void copy(const HyperParameter& trg);

		// accessor
		int getNLayers() const { return num_layers; }
		int getNNeurons(const int& layer) const { return num_neurons[layer]; }
		
		// mutator
		void setNNeurons(const int& layer, const int& value) { num_neurons[layer] = value; }

		// Hyper Parameter
		R learning_rate;

	private:
		int num_layers; // number of layers (input layer + hidden layer + output layer)
		int* num_neurons;
	};

	class NeuralNetwork {
	public:
		NeuralNetwork() : layer(nullptr), num_input(0), num_layers(0) {}
		NeuralNetwork(const HyperParameter& hp)
			: hp(hp), layer(nullptr), num_input(0), num_layers(0) { initialize(); }
		virtual ~NeuralNetwork() { Layer_Delete(); }
		
		void initialize();
		void initialize(const HyperParameter& hp) { NeuralNetwork::hp = hp; initialize(); }

		R learning(const ExMatrix& input, const ExMatrix& label, const bool getCost = false);
		const ExMatrix activate(const ExMatrix& input);

	protected:
		HyperParameter hp;
		Layer** layer;
		int num_input, num_output, num_layers;

		// initialize
		void Layer_Delete();
		virtual void Layer_Initialize();
		virtual void Weight_Initialize();

		// main function
		void Forward_Propagation(const ExMatrix& input);
		void Backward_Propagation(const ExMatrix& label);
		void Weight_Update();
		const ExMatrix Loss_Function(const ExMatrix& yhat, const ExMatrix& y) const;
		R Cost_Function(const ExMatrix& label) const;
	};
}