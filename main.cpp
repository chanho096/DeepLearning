//#include "Test_FNNInitialize.h"
#include "NeuralNetwork.h"
#include "Test_FNNMnist.h"
#include <ctime>
#include <iostream>
using std::cout; using std::endl;
using fnn::NeuralNetwork;
using fnn::HyperParameter;

int main() {
	ExMatrix train_input, train_label, test_input, test_label;
	ExMatrix result, trs;
	const int labelSize = 10; // 0 ~ 9
	int imageSize, num_train, num_test;
	R maxval, maxnum, labelnum;
	
	// data update
	getImage(train_input, "train-images.idx3-ubyte");
	getLabel(train_label, "train-labels.idx1-ubyte", labelSize);

	getImage(test_input, "t10k-images.idx3-ubyte");
	getLabel(test_label, "t10k-labels.idx1-ubyte", labelSize);
	imageSize = train_input.getNRows();
	num_train = train_input.getNColumns();
	num_test = test_input.getNColumns();

	cout << num_train << " train data and " << num_test << " test data updated." << endl;
	cout << "image has " << imageSize << " pixels in each." << endl;
	system("pause");
	system("cls");

	// data normialize
	train_input.multiply(1 / 255);
	test_input.multiply(1 / 255);

	// parameters
	int num_layers = 3;
	int num_neurons = 8;
	int num_epoch = 100;

	// initialize
	srand((unsigned int)time(NULL));
	HyperParameter hp;
	hp.initialize(num_layers);
	for (int i = 1; i < num_layers - 1; ++i)
		hp.setNNeurons(i, num_neurons);
	hp.setNNeurons(0, imageSize);
	hp.setNNeurons(num_layers - 1, labelSize);
	hp.learning_rate = 0.05;
	NeuralNetwork* fnn = new NeuralNetwork(hp);
	
	// train
	for (int i = 0; i < num_epoch; ++i) {
		if (i % 10 == 0)
			cout << fnn->learning(test_input, test_label, true) << endl;
		else
			fnn->learning(test_input, test_label);
	}
	cout << "the training of " << num_epoch << " epoch has been completed." << endl;
	system("pause");
	system("cls");

	R train_ratio = 0.0;
	result = fnn->activate(test_input);
	trs.initialize(result.getNRows(), result.getNColumns());
	for (int i = 0; i < num_test; ++i) {
		// argmax output
		maxval = result(0, i);
		maxnum = 0;
		for (int j = 1; j < labelSize; ++j) {
			if (maxval < result(j, i)) {
				maxval = result(j, i);
				maxnum = j;
			}
			if (train_label(j, i) == (R)1)
				labelnum = j;
		}

		if (maxnum == labelnum)
			train_ratio = train_ratio + (R)1;
	}
	train_ratio /= num_train;
	cout << train_ratio * 100 << " % accurate in train set." << endl;
	system("pause");
	// test
	R test_ratio = 0.0;
	result = fnn->activate(test_input);
	trs.initialize(result.getNRows(), result.getNColumns());
	for (int i = 0; i < num_test; ++i) {
		// argmax output
		maxval = result(0, i);
		maxnum = 0;
		for (int j = 1; j < labelSize; ++j) {
			if (maxval < result(j, i)) {
				maxval = result(j, i);
				maxnum = j;
			}
			if (test_label(j, i) == (R)1)
				labelnum = j;
		}

		if (maxnum == labelnum)
			test_ratio = test_ratio + (R)1;
	}
	test_ratio /= num_test;
	cout << test_ratio * 100 << " % accurate in test set." << endl;


	system("pause");

	return 0;
}

