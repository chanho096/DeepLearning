#include "Test_FNNInitialize.h"
#include "Test_Matrix.h"

using std::cout; using std::endl; using std::ostream;
using fnn::Layer; using namespace actf;
using alg::Matrix; using alg::ExMatrix;

ostream& operator << (ostream& os, ActfType actf) {
	switch (actf) {
	case ActfType::TSigmoid:
		os << "Sigmoid";
		break;
	case ActfType::TReLU:
		os << "ReLU";
		break;
	default:
		os << "Unknown";
	}
	return os;
}

int Test_FNN() {
	HyperParameter hp;
	hp.initialize(3);
	hp.learning_rate = 0.01;
	hp.setNNeurons(0, 5);
	hp.setNNeurons(1, 3);
	hp.setNNeurons(2, 5);

	cout << "First FNN Initialize Test" << endl;
	system("pause");
	FNN_Test* tstfnn = new FNN_Test(hp);
	tstfnn->test_layerInit();
	delete tstfnn; tstfnn = nullptr;

	system("cls");
	cout << "Second FNN Initialize Test" << endl;
	system("pause");

	tstfnn = (FNN_Test*)new FNN_Test_Sigmoid(hp);
	tstfnn->test_layerInit();

	system("cls");
	cout << "FNN Rebuild Test" << endl;
	system("pause");

	hp.setNNeurons(0, 10);
	hp.setNNeurons(1, 10);
	hp.setNNeurons(2, 10);
	tstfnn->initialize(hp);
	tstfnn->test_layerInit();
	
	return 0;
}

void FNN_Test::test_layerInit() {
	ExMatrix tstw;
	cout.precision(3);

	for (int i = 0; i < num_layers; ++i) {
		system("cls");
		cout << "FNN Test - Layer Initialize" << endl;

		// Layer Number
		cout << "Layer " << i; 
		if (i == 0) cout << " (Input Layer)";
		else if (i == num_layers - 1) cout << " (Output Layer)";
		cout << " : " << endl;

		// Layer Information
		cout << "- Activation Function Type : " << layer[i]->getActf() << endl;
		cout << "- Number Of Neuron : " << layer[i]->getNNeurons() << endl;
		cout << " (Number Of Neuron in HyperParameter : " << hp.getNNeurons(i) << ")" << endl;
		cout << endl;

		// Weight
		if (i == 0) {
			cout << "no weight in input layer (dummy layer)" << endl;
			system("pause");
			continue;
		}
		tstw = layer[i]->getWeight();
		cout << "Weight : " << tstw << endl << endl;
		test_weightInit(i);

		cout << "real average : " << tstw.average() << endl;
		cout << "real variance : " << tstw.variance() << endl;
		cout << "( fan_in : " << layer[i]->getNNeurons() << ", fan_out : ";
		if (i != num_layers - 1)
			cout << layer[i + 1]->getNNeurons();
		else
			cout << "0";
		cout << " )" << endl;
		system("pause");
	}
}

void FNN_Test::test_weightInit(const int &layer){
	assert(layer > 0);
	R fan_in = NeuralNetwork::layer[layer]->getNNeurons();

	cout << "this FNN using ReLU activation function / He-Initialization" << endl;
	cout << "(output layer has Sigmoid activation function)" << endl << endl;

	cout << "min value : " << -1 * sqrt(6 / fan_in) << ", max value : " << sqrt(6 / fan_in) << endl;
	cout << "theorical average : 0.00" << endl;
	cout << "theorical variance : " << 2 / fan_in << endl;
}

void FNN_Test_Sigmoid::test_weightInit(const int &layer) {
	assert(layer > 0);
	R fan_in = NeuralNetwork::layer[layer]->getNNeurons();
	R fan_out = 0;
	if (layer < num_layers - 1) fan_out = NeuralNetwork::layer[layer + 1]->getNNeurons();

	cout << "this FNN using Sigmoid activation function / Xavier-Initialization" << endl;

	cout << "min value : " << -1 * sqrt(6 / (fan_in + fan_out)) 
		<< ", max value : " << sqrt(6 / (fan_in + fan_out)) << endl;
	cout << "theorical average : 0.00" << endl;
	cout << "theorical variance : " << 2 / (fan_in + fan_out) << endl;
}

void FNN_Test_Sigmoid::Layer_Initialize() {
	// activation function : Sigmoid
	for (int i = 0; i < num_layers; ++i) layer[i]->setActf(ActfType::TSigmoid);
}

void FNN_Test_Sigmoid::Weight_Initialize() {
	// Use Xavier Initialization
	for (int i = 1; i < num_layers - 1; ++i) layer[i]->Xavier_Initialize(hp.getNNeurons(i), hp.getNNeurons(i + 1));
	layer[num_layers - 1]->Xavier_Initialize(hp.getNNeurons(num_layers - 1), 0);
}
