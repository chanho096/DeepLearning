//#include "Test_FNNInitialize.h"
#include "NeuralNetwork.h"
#include "Test_FNNInitialize.h"
#include <iostream>
#include <ctime>
using fnn::NeuralNetwork; using fnn::HyperParameter;
using alg::ExMatrix;
using std::cout; using std::endl;

int main() {
	const int num_layers = 5;
	const int num_neurons = 8;

	const int num_train = 1000;
	const int num_test = 10000;
	const int num_epoch = 1000;
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
	//for (int k = 0; k < 100; ++k) {

		// Train
		fnn = new NeuralNetwork(hp);
		for (int i = 0; i < num_epoch; ++i) {
			if (i % 10 == 0)
				cout << fnn->learning(train_input, train_label, true) << endl;
			else
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
		//cout << k << " seed : ";
		cout << ratio_test * 100 << " % acuurate in test / ";
		cout << ratio_train * 100 << " % acuurate in train";
		
		if (ratio_test < 0.7)
			cout << " (fail) ";
		cout << endl;

		//system("pause");

		delete fnn;
	//}
	system("pause");
	return 0;
}
//