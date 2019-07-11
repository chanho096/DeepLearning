#pragma once
#include <iostream>
#include "ExtendedMatrix.h"

using std::cout; using std::ostream; using std::endl;
using namespace alg;
int Test_Matrix();

ostream& operator << (ostream& os, const Matrix& trg);
void arrangeMatrix(Matrix& trg);