#include "Matrix.h"
#include <math.h>

namespace alg {
	void Matrix::initialize() {
		assert(values != nullptr);
		for (int i = 0; i < size; ++i) {
			values[i] = (R)0.0;
		}
	}

	void Matrix::initialize(const int& m, const int& n, const bool init) {
		assert(m > 0 && n > 0);
		const int old_size = size;
		num_rows = m; num_cols = n;
		size = num_rows * num_cols;

		if (old_size != size) {
			SAFE_DELETE_ARRAY(values);
			values = new R[size];
		}

		if (init == true) {
			initialize();
		}
	}

	const R& Matrix::operator () (const int & m, const int & n) const
	{
		assert(m >= 0 && m < num_rows);
		assert(n >= 0 && n < num_cols);
		return values[m * num_cols + n];
	}

	R& Matrix::operator () (const int& m, const int& n) {
		assert(m >= 0 && m < num_rows);
		assert(n >= 0 && n < num_cols);
		return values[m * num_cols + n];
	}

	void Matrix::copy(const Matrix& trg) {
		if (trg.num_rows != num_rows || trg.num_cols != num_cols) {
			initialize(trg.num_rows, trg.num_cols, false);
		}

		for (int i = 0; i < size; ++i) {
			values[i] = trg.values[i];
		}
	}

	void Matrix::reverse() {
		for (int i = 0; i < size; ++i)
			values[i] = (R)1.0 / values[i];
	}

	void Matrix::transpose() {
		int i, j;
		Matrix tmp(num_cols, num_rows);
		for (j = 0; j < num_cols; ++j) {
			for (i = 0; i < num_rows; ++i) {
				tmp.values[j * tmp.num_cols + i] = values[i * num_cols + j];
			}
		}
		copy(tmp);
	}

	void Matrix::sum(Matrix& result, int axis) const {
		int row, col, idx;
		if (axis == 0) {
			// vertical summation
			result.initialize(1, num_cols);
			for (row = 0; row < num_rows; ++row) {
				idx = row * num_cols;
				for (col = 0; col < num_cols; ++col) {
					result.values[col] += values[idx + col];
				}
			}
		}
		else if (axis == 1) {
			// horizontal summation
			result.initialize(num_rows, 1);
			for (row = 0; row < num_rows; ++row) {
				idx = row * num_cols;
				for (col = 0; col < num_cols; ++col) {
					result.values[row] += values[idx + col];
				}
			}
		}
	}

	void Matrix::product(const Matrix& trg, Matrix& result) const {
		int row, col, dim, idx1, idx2, tmp;
		result.initialize(num_rows, trg.num_cols, false);
		assert(num_cols == trg.num_rows);

		for (row = 0; row < result.num_rows; ++row) {
			idx1 = row * result.num_cols;
			tmp = row * num_cols;

			for (col = 0; col < result.num_cols; ++col) {
				idx2 = idx1 + col;
				result.values[idx2] = (R)0.0;

				for (dim = 0; dim < num_cols; ++dim) {
					result.values[idx2] += values[tmp + dim] * trg.values[col + dim * trg.num_cols];
				}
			}
		}
	}

	void Matrix::productTransposed(const Matrix& trg, Matrix& result) const {
		int row, col, dim, idx1, idx2, tmp1, tmp2;
		result.initialize(num_rows, trg.num_rows, false);
		assert(num_cols == trg.num_cols);

		for (row = 0; row < result.num_rows; ++row) {
			idx1 = row * result.num_cols;
			tmp1 = row * num_cols;
			for (col = 0; col < result.num_cols; ++col) {
				idx2 = idx1 + col;
				tmp2 = col * trg.num_cols;
				result.values[idx2] = (R)0.0;

				for (dim = 0; dim < num_cols; ++dim) {
					result.values[idx2] += values[tmp1 + dim] * trg.values[tmp2 + dim];
				}
			}
		}
	}

	void Matrix::productWithTranspose(const Matrix &trg, Matrix & result) const {
		int row, col, dim, idx1, idx2;
		result.initialize(num_cols, trg.num_cols, false);
		assert(num_rows == trg.num_rows);

		for (row = 0; row < result.num_rows; ++row) {
			idx1 = row * result.num_cols;
			for (col = 0; col < result.num_cols; ++col) {
				idx2 = idx1 + col;
				result.values[idx2] = (R)0.0;

				for (dim = 0; dim < num_rows; ++dim) {
					result.values[idx2] += values[row + dim * num_cols] * trg.values[col + dim * trg.num_cols];
				}
			}
		}
	}

	void Matrix::multiply(const Matrix& trg) {
		if (num_rows == trg.num_rows && num_cols == trg.num_cols) {
			for (int i = 0; i < size; ++i) {
				values[i] *= trg.values[i];
			}
		}
		else if (trg.num_rows == 1 && num_cols == trg.num_cols) {
			// element-wise multiply with row vector
			multiplyVector(trg, 0);
		}
		else if (trg.num_cols == 1 && num_rows == trg.num_rows) {
			// element-wise multiply with column vector
			multiplyVector(trg, 1);
		}
	}

	void Matrix::multiply(const R& trg) {
		for (int i = 0; i < size; ++i)
			values[i] *= trg;
	}

	void Matrix::addition(const Matrix& trg) {
		if (num_rows == trg.num_rows && num_cols == trg.num_cols) {
			for (int i = 0; i < size; ++i) {
				values[i] += trg.values[i];
			}
		}
		else if (trg.num_rows == 1 && num_cols == trg.num_cols) {
			// element-wise multiply with row vector
			additionVector(trg, 0);
		}
		else if (trg.num_cols == 1 && num_rows == trg.num_rows) {
			// element-wise multiply with column vector
			additionVector(trg, 1);
		}
	}

	void Matrix::addition(const R& trg) {
		for (int i = 0; i < size; ++i)
			values[i] += trg;
	}

	void Matrix::multiplyVector(const Matrix& trg, const int& axis) {
		int row, col, idx;
		if (axis == 0) {
			assert(trg.num_rows == 1 && num_cols == trg.num_cols);
			for (row = 0; row < num_rows; ++row) {
				idx = row * num_cols;
				for (col = 0; col < num_cols; ++col) {
					values[idx + col] *= trg.values[col];
				}
			}
		}

		else if (axis == 1) {
			assert(trg.num_cols == 1 && num_rows == trg.num_rows);
			for (row = 0; row < num_rows; ++row) {
				idx = row * num_cols;
				for (col = 0; col < num_cols; ++col) {
					values[idx + col] *= trg.values[row];
				}
			}
		}
	}

	void Matrix::additionVector(const Matrix& trg, const int& axis) {
		int row, col, idx;
		if (axis == 0) {
			assert(trg.num_rows == 1 && num_cols == trg.num_cols);
			for (row = 0; row < num_rows; ++row) {
				idx = row * num_cols;
				for (col = 0; col < num_cols; ++col) {
					values[idx + col] += trg.values[col];
				}
			}
		}

		else if (axis == 1) {
			assert(trg.num_cols == 1 && num_rows == trg.num_rows);
			for (row = 0; row < num_rows; ++row) {
				idx = row * num_cols;
				for (col = 0; col < num_cols; ++col) {
					values[idx + col] += trg.values[row];
				}
			}
		}
	}

	const Matrix Matrix::operator < (const R& trg) const {
		Matrix tmp;
		tmp.initialize(num_rows, num_cols, false);
		for (int i = 0; i < size; i++) {
			if (values[i] < trg)
				tmp.values[i] = (R)true;
			else
				tmp.values[i] = (R)false;
		}
		return tmp;
	}

	const Matrix Matrix::operator <= (const R& trg) const {
		Matrix tmp;
		tmp.initialize(num_rows, num_cols, false);
		for (int i = 0; i <= size; i++) {
			if (values[i] <= trg)
				tmp.values[i] = (R)true;
			else
				tmp.values[i] = (R)false;
		}
		return tmp;
	}

	const Matrix Matrix::operator > (const R& trg) const {
		Matrix tmp;
		tmp.initialize(num_rows, num_cols, false);
		for (int i = 0; i < size; i++) {
			if (values[i] > trg)
				tmp.values[i] = (R)true;
			else
				tmp.values[i] = (R)false;
		}
		return tmp;
	}

	const Matrix Matrix::operator >= (const R& trg) const {
		Matrix tmp;
		tmp.initialize(num_rows, num_cols, false);
		for (int i = 0; i < size; i++) {
			if (values[i] >= trg)
				tmp.values[i] = (R)true;
			else
				tmp.values[i] = (R)false;
		}
		return tmp;
	}

	Matrix& Matrix::operator = (const R& trg) {
		for (int i = 0; i < size; ++i) {
			values[i] = trg;
		}
		return *this;
	}
}