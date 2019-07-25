#pragma once
#include "Matrix.h"
#include <random>

namespace alg {
	// Extended Matrix for Feed-Forward Neural Network
	class ExMatrix : public Matrix {
	public:
		ExMatrix() : Matrix() {}
		ExMatrix(const int& num_rows, const int& num_cols) : Matrix(num_rows, num_cols) {}
		ExMatrix(const ExMatrix& trg) : Matrix() { copy(trg); }
		ExMatrix(const Matrix& trg) : Matrix(trg) {}
		virtual ~ExMatrix() {}

		ExMatrix& operator = (const ExMatrix& trg) { copy(trg); return *this; }
		ExMatrix& operator = (const Matrix& trg) { Matrix::copy(trg); return *this; }
		void copy(const ExMatrix& trg) { Matrix::copy((Matrix)trg); }

		void randomize();
		void randomize(const R& scale, const R& min);
		void normalize(const R& min, const int& aixs);
		
		R average() const;
		R variance() const;

		void ReLU();
		void dReLU();
		void Sigmoid();
		void dSigmoid();

		// math library
		void log();
		void exp();
	
	private:
		// random library
		typedef std::default_random_engine engine;
		typedef std::normal_distribution<R> dis_normal;
		typedef std::uniform_real_distribution<R> dis_uniform;
		static std::random_device rd;
		static engine generator;
		
	};
}