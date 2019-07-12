#include "NeuralNetwork.h"
#include <cstdlib>
#include <time.h>

namespace fnn {
	using actf::ActfType;
	void HyperParameter::initialize() {
		SAFE_DELETE_ARRAY(num_neurons);
		num_layers = 0;

		// Hyper Parameter
		learning_rate = 0.0;
	}

	void HyperParameter::initialize(const int& num_layers) {
		initialize();
		HyperParameter::num_layers = num_layers;
		num_neurons = new int[num_layers];
		for (int i = 0; i < num_layers; ++i)
			num_neurons[i] = 0;
	}
	void HyperParameter::copy(const HyperParameter &trg) {
		if (num_layers != trg.num_layers) {
			SAFE_DELETE_ARRAY(num_neurons);
			num_layers = trg.num_layers;
			if (trg.num_layers != 0) {
				assert(trg.num_neurons != nullptr);
				num_neurons = new int[num_layers];
			}
		}
		for (int i = 0; i < num_layers; ++i)
			num_neurons[i] = trg.num_neurons[i];

		// Hyper Parameter
		learning_rate = trg.learning_rate;
	}

	void NeuralNetwork::initialize() {
		assert(hp.num_layers > 1);
		// Data Update
		NeuralNetwork::num_input = hp.num_neurons[0];
		NeuralNetwork::num_layers = hp.num_layers;
		
		// Creating Layer
		Layer_Delete();
		layer = new Layer*[num_layers];
		for (int i = 0; i < num_layers; ++i) layer[i] = new Layer();
		for (int i = 1; i < num_layers; ++i) layer[i]->rebuild(hp.num_neurons[i], hp.num_neurons[i - 1]);

		// Initialize
		srand((unsigned int)time(NULL));
		Layer_Initialize();
		Weight_Initialize();
	}

	void NeuralNetwork::Layer_Delete() {
		if (layer == nullptr) return;
		for (int i = 0; i < hp.num_layers; ++i) SAFE_DELETE(layer[i]);
		SAFE_DELETE_ARRAY(layer);
	}

	void NeuralNetwork::Layer_Initialize() {
		// basic activate function : ReLU
		for (int i = 0; i < num_layers - 1; ++i) layer[i]->setActf(ActfType::TReLU);

		// output activate function : Sigmoid
		layer[num_layers - 1]->setActf(ActfType::TSigmoid);
	}

	void NeuralNetwork::Weight_Initialize() {
		// Use He Initialization for ReLU activation
		for (int i = 1; i < num_layers; ++i) layer[i]->He_Initialize(hp.num_neurons[i]);
	}
	
}
