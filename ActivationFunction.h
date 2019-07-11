#pragma once
#include "Common.h"

namespace actf {
	typedef enum { TSigmoid = 0, TReLU } ActfType;
	R ReLU(const R& x);
	R dReLU(const R& x);

	R Sigmoid(const R& x);
	R dSigmoid(const R& y);
}