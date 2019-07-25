#pragma once
#include "NeuralNetwork.h"
#include <string>

using alg::ExMatrix;
using std::string;

int reverseInt(int i);
void getImage(ExMatrix& result, const string& path);
void getLabel(ExMatrix& result, const string& path, const int& labelSize);