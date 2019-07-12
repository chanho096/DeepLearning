#pragma once
#include "Test_Matrix.h"
#include "NeuralNetwork.h"

using fnn::HyperParameter;
using fnn::NeuralNetwork;

std::ostream& operator << (std::ostream&, actf::ActfType);
int Test_FNN();

// Using ReLU / He Initialize
class FNN_Test : public NeuralNetwork {
public:
	FNN_Test() : NeuralNetwork() {}
	FNN_Test(const HyperParameter& hp) : NeuralNetwork() { initialize(hp); }
	virtual ~FNN_Test() {}
	void test_layerInit();

protected:
	virtual void test_weightInit(const int& layer);
};

// Using Sigmoid / Xavier Initialize 
class FNN_Test_Sigmoid : public FNN_Test {
public:
	FNN_Test_Sigmoid(const HyperParameter& hp) : FNN_Test() { initialize(hp); }
	virtual ~FNN_Test_Sigmoid() {}

protected:
	void test_weightInit(const int& layer);
	void Layer_Initialize();
	void Weight_Initialize();
};