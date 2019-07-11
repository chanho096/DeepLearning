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
		void setNNeurons(const int& value, const int& layer) { num_neurons[layer] = value; }

		// Hyper Parameter
		R learning_rate;

	private:
		int num_layers; // number of layers (input layer + hidden layer + 
		int* num_neurons;
	};

	class NeuralNetwork {
	public:
		NeuralNetwork(const HyperParameter& hp, const int& num_input)
			: hp(hp), num_input(num_input), layer(nullptr) { initialize(); };
		virtual ~NeuralNetwork() { Layer_Delete(); }
		
		void initialize();
		void initialize(const HyperParameter& hp) { NeuralNetwork::hp = hp; initialize(); }

	protected:
		HyperParameter hp;
		Layer** layer;
		int num_input;

		// initialize
		void Layer_Delete();
		virtual void Layer_Initialize();
		virtual void Weight_Initialize();

		// main function
		void Forward_Propagation();
		void Backward_Propagation();
	};
}