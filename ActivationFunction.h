#pragma once
#include "Common.h"

namespace actf {
	typedef enum { TSigmoid = 0, TReLU, TSoftmax, TLReLU } ActfType;

	R ReLU(const R& x);
	R dReLU(const R& x);

	R Sigmoid(const R& x);
	R dSigmoid(const R& y);

	R LReLU(const R& x);
	R dLReLU(const R& x);
}