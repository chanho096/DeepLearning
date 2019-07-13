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
		assert(hp.num_layers > 2);
		// data update
		NeuralNetwork::num_input = hp.num_neurons[0];
		NeuralNetwork::num_output = hp.num_neurons[hp.num_layers - 1];
		NeuralNetwork::num_layers = hp.num_layers;
		
		// creating layer
		Layer_Delete();
		layer = new Layer*[num_layers];
		for (int i = 0; i < num_layers; ++i) layer[i] = new Layer();
		for (int i = 1; i < num_layers; ++i) layer[i]->rebuild(hp.num_neurons[i], hp.num_neurons[i - 1]);

		// layer Initialize
		srand((unsigned int)time(NULL));
		Layer_Initialize();

		// output activation function : Sigmoid
		layer[num_layers - 1]->setActf(ActfType::TSigmoid);

		// weight Initialize
		Weight_Initialize();
	}

	void NeuralNetwork::learning(const ExMatrix &input, const ExMatrix &lable) {
		assert(num_input == input.getNRows());
		assert(num_output == lable.getNRows());
		assert(input.getNColumns() == lable.getNColumns());

		Forward_Propagation(input);
		Backward_Propagation(lable);
		Weight_Update(); // gradient descent

	}

	void NeuralNetwork::Layer_Delete() {
		if (layer == nullptr) return;
		for (int i = 0; i < hp.num_layers; ++i) SAFE_DELETE(layer[i]);
		SAFE_DELETE_ARRAY(layer);
	}

	void NeuralNetwork::Layer_Initialize() {
		// basic activation function : ReLU
		for (int i = 0; i < num_layers - 1; ++i) layer[i]->setActf(ActfType::TReLU);
	}

	void NeuralNetwork::Weight_Initialize() {
		// Use He Initialization for ReLU activation function
		for (int i = 1; i < num_layers; ++i) layer[i]->He_Initialize(hp.num_neurons[i]);
	}

	void NeuralNetwork::Forward_Propagation(const ExMatrix &input) {
		layer[0]->Set_Input(input);
		for (int i = 1; i < num_layers; ++i)
			layer[i]->Forward_Propagation(layer[i - 1]);
	}

	void NeuralNetwork::Backward_Propagation(const ExMatrix& lable) {
		assert(num_layers > 2);
		// Cross-Entropy loss function and Sigmoid (in output layer) are assumed.
		layer[num_layers - 1]->Set_Gradient(lable, layer[num_layers - 2]);
		for (int i = num_layers - 2; i > 0; ++i)
			layer[i]->Backward_Propagation(layer[i - 1], layer[i + 1]);
	}
	
	void NeuralNetwork::Weight_Update() {
		for (int i = 1; i < num_layers; ++i)
			layer[i]->Weight_Update(hp.learning_rate);
	}
}
