#include "Test_FNNInitialize.h"
#include "Test_Matrix.h"
#include <ctime>

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

int Test_FNN_Learning() {
	const int num_layers = 3;
	const int num_neurons = 8;

	const int num_train = 1000;
	const int num_test = 10000;
	const int num_epoch = 3000;
	const R threshold = 0.5;

	ExMatrix train_input(1, num_train), train_label(2, num_train);
	ExMatrix test_input(1, num_test), test_label(2, num_test);
	ExMatrix result, trs;
	HyperParameter hp;
	NeuralNetwork* fnn;
	R ratio_test, ratio_train;

	//system("cls");
	srand((unsigned int)0);

	// Initialize
	hp.initialize(num_layers);
	hp.learning_rate = 0.01;

	hp.setNNeurons(0, 1);
	for (int i = 1; i < num_layers - 1; ++i) hp.setNNeurons(i, num_neurons);
	hp.setNNeurons(num_layers - 1, 2);

	train_input.randomize(); test_input.randomize();
	for (int i = 0; i < num_train; ++i) {
		if (train_input(0, i) < threshold) {
			train_label(0, i) = 1;
			train_label(1, i) = 0;
		}
		else {
			train_label(0, i) = 0;
			train_label(1, i) = 1;
		}
	}

	for (int i = 0; i < num_test; ++i) {
		if (test_input(0, i) < threshold) {
			test_label(0, i) = 1;
			test_label(1, i) = 0;
		}
		else {
			test_label(0, i) = 0;
			test_label(1, i) = 1;
		}
	}

	srand((unsigned int)time(NULL));
	for (int k = 0; k < 100; ++k) {

		// Train
	fnn = new NeuralNetwork(hp);
	for (int i = 0; i < num_epoch; ++i) {
		//if (i % 10 == 0)
		//	cout << fnn->learning(train_input, train_label, true) << endl;
		//else
			fnn->learning(train_input, train_label);
	}

	//cout << "Training Complete" << endl;
	//system("pause");
	//system("cls");

	// Train
	result = fnn->activate(train_input);
	trs.initialize(2, num_train);
	for (int i = 0; i < num_train; ++i) {
		if (result(0, i) > result(1, i)) {
			trs(0, i) = 1;
			trs(1, i) = 0;
		}
		else {
			trs(0, i) = 0;
			trs(1, i) = 1;
		}
	}

	ratio_train = 0;
	for (int i = 0; i < num_train; ++i) {
		if (trs(0, i) == train_label(0, i))
			ratio_train = ratio_train + 1;
	}
	ratio_train = ratio_train / num_train;

	// Test
	//cout << "Test with " << num_train << " train samples" << endl << endl;
	result = fnn->activate(test_input);
	trs.initialize(2, num_test);
	for (int i = 0; i < num_test; ++i) {
		if (result(0, i) > result(1, i)) {
			trs(0, i) = 1;
			trs(1, i) = 0;
		}
		else {
			trs(0, i) = 0;
			trs(1, i) = 1;
		}
	}

	ratio_test = 0;
	for (int i = 0; i < num_test; ++i) {
		if (trs(0, i) == test_label(0, i))
			ratio_test = ratio_test + 1;
	}
	ratio_test = ratio_test / num_test;


	//cout << "Train Input Matrix : " << endl << train_input << endl << endl;
	//cout << "Test Input Matrix : " << endl << test_input << endl << endl;
	//cout << "Labeled Matrix : ( threshold 0.5 )" << endl << test_label << endl << endl;
	//cout << "Activated Matrix : " << endl << result << endl << endl;
	//cout << "Transed Matrix :" << endl << trs << endl << endl;
	cout << k << " seed : ";
	cout << ratio_test * 100 << " % acuurate in test / ";
	cout << ratio_train * 100 << " % acuurate in train";

	if (ratio_test < 0.7)
		cout << " (fail) ";
	cout << endl;

	//system("pause");

	delete fnn;
	}
	system("pause");
	return 0;
}