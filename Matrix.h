#pragma once
#include <assert.h>
#include "Common.h"

namespace alg {
	class Matrix {
	public:
		Matrix()
			: num_rows(0), num_cols(0), size(0), values(nullptr) {}

		Matrix(const int& num_rows, const int& num_cols)
			: size(0), values(nullptr) {
			initialize(num_rows, num_cols);
		}

		Matrix(const Matrix& trg)
			: size(0), values(nullptr) {
			copy(trg);
		}

		virtual ~Matrix() { SAFE_DELETE_ARRAY(values); num_rows = 0; num_cols = 0; }

		// assignment operator overloading
		Matrix& operator = (const Matrix& trg) { copy(trg); return *this; }
		Matrix& operator = (const R& trg);
		
		// initialize
		void initialize();
		void initialize(const int& m, const int& n, const bool init = true);

		// accessor
		int getSize() const { return size; }
		int getNRows() const { return num_rows; }
		int getNColumns() const { return num_cols; }
		const R& operator () (const int& m, const int&n) const;

		// basic function
		R& operator () (const int& m, const int& n);
		void copy(const Matrix& trg);
		void reverse();
		void transpose();
		void sum(Matrix& result, int axis) const;

		// operation
		void product(const Matrix& trg, Matrix& result) const;
		void productTransposed(const Matrix& trg, Matrix& result) const;
		void multiply(const Matrix& trg);
		void multiply(const R& trg);
		void addition(const Matrix& trg);
		void addition(const R& trg);

		// threshold
		const Matrix operator < (const R& trg) const;
		const Matrix operator <= (const R& trg) const;
		const Matrix operator > (const R& trg) const;
		const Matrix operator >= (const R& trg) const;

	protected:
		int num_rows, num_cols, size;
		R* values;

		void multiplyVector(const Matrix& trg, const int& axis);
		void additionVector(const Matrix& trg, const int& axis);
	};
}