#include "Test_Matrix.h"

int Test_Matrix() {
	Matrix mat1, mat2, mat3, vec1, vec2;
	
	cout << " [ 1 ]" << endl;
	cout << "* Creating Matrix 1, Matrix 2" << endl;
	mat1.initialize(3, 2); arrangeMatrix(mat1);
	mat2.initialize(2, 3); arrangeMatrix(mat2);

	cout << "Matrix 1: " << endl << mat1 << endl;
	cout << "Matrix 2: " << endl << mat2 << endl;

	system("pause");
	system("cls");
	cout << " [ 2 ]" << endl;

	cout << "* Matrix 1 = Matrix 1 * 10" << endl;
	mat1.multiply(10);
	cout << "Matrix 1: " << endl << mat1 << endl;

	cout << "* Matrix 2 = Matrix 2 + 10" << endl;
	mat2.addition(10);
	cout << "Matrix 2: " << endl << mat2 << endl;

	system("pause");
	system("cls");
	cout << " [ 3 ]" << endl;

	cout << "* Matrix 3 = Matrix 1 * Matrix 2 (matrix product)" << endl;
	mat1.product(mat2, mat3);
	cout << "Matrix 1: " << endl << mat1 << endl;
	cout << "Matrix 2: " << endl << mat2 << endl;
	cout << "Matrix 3: " << endl << mat3 << endl;

	system("pause");
	system("cls");
	cout << " [ 4 ]" << endl;

	cout << "* Matrix 3 Reversed" << endl;
	arrangeMatrix(mat3);
	cout << "Matrix 3: " << endl << mat3 << endl;
	mat3.reverse();
	cout << "Matrix 3: " << endl << mat3 << endl;

	cout << "* Matrix 2 Transposed" << endl;
	cout << "Matrix 2: " << endl << mat2 << endl;
	mat2.transpose();
	cout << "Matrix 2: " << endl << mat2 << endl;


	system("pause");
	system("cls");
	cout << " [ 5 ]" << endl;
	cout << "* Matrix 3 = Matrix 1 * Matrix 2 (matrix transposed product)" << endl;
	mat1.productTransposed(mat2, mat3);
	cout << "Matrix 1: " << endl << mat1 << endl;
	cout << "Matrix 2: " << endl << mat2 << endl;
	cout << "Matrix 3: " << endl << mat3 << endl;

	system("pause");
	system("cls");
	cout << " [ 6 ]" << endl;

	cout << "Matrix 1: " << endl << mat1 << endl;
	cout << "Matrix 2: " << endl << mat2 << endl;

	cout << "* Matrix 1 = Matrix 1 + Matrix 2" << endl;
	mat1.addition(mat2);
	cout << "Matrix 1: " << endl << mat1 << endl;

	cout << "* Matrix 2 = Matrix 1 * Matrix 2 (matrix multiply)" << endl;
	mat2.multiply(mat1);
	cout << "Matrix 2: " << endl << mat2 << endl;

	system("pause");
	system("cls");
	cout << " [ 7 ]" << endl;
	cout << "Matrix 1: " << endl << mat1 << endl;
	
	cout << "* Vector 1 = Horizontal Summation of Matrix 1" << endl;
	mat1.sum(vec1, 1);
	cout << "Vector 1: " << endl << vec1 << endl;

	cout << "* Vector 2 = Vertical Summation of Matrix 1" << endl;
	mat1.sum(vec2, 0);
	cout << "Vector 2: " << endl << vec2 << endl;

	system("pause");
	system("cls");
	cout << " [ 8 ]" << endl;

	cout << "* Matrix 1 = Matrix 1 * Vector 1 (element-wise multiply)" << endl;
	cout << "Matrix 1: " << endl << mat1 << endl;
	cout << "Vector 1: " << endl << vec1 << endl;
	mat1.multiply(vec1);
	cout << "Matrix 1: " << endl << mat1 << endl;

	system("pause");
	system("cls");
	cout << " [ 9 ]" << endl;

	cout << "* Matrix 2 = Matrix 2 + Vector 2 (element-wise addition)" << endl;
	cout << "Matrix 2: " << endl << mat2 << endl;
	cout << "Vector 2: " << endl << vec2 << endl;
	mat2.addition(vec2);
	cout << "Matrix 2: " << endl << mat2 << endl;

	system("pause");
	system("cls");
	cout << " [ 10 ]" << endl;
	cout << "* Matrix 3 = Matrix 1 < 5000 (element-wise threshold)" << endl;
	mat3 = mat1 < 5000;
	cout << "Matrix 1: " << endl << mat1 << endl;
	cout << "Matrix 3: " << endl << mat3 << endl;

	cout << "* Matrix 3 = Matrix 2 >= 990 (element-wise threshold)" << endl;
	mat3 = mat2 >= 990;
	cout << "Matrix 2: " << endl << mat2 << endl;
	cout << "Matrix 3: " << endl << mat3 << endl;
	
	system("pause");
	system("cls");

	cout << " [ End Of Test ]" << endl;
	system("pause");

	return 0;
}

ostream& operator << (ostream& os, const Matrix& trg) {
	int i, j, row, col;
	row = trg.getNRows(); col = trg.getNColumns();
	for (i = 0; i < row; ++i) {
		os << "[ ";
		for (j = 0; j < col; ++j) {
			os << trg(i, j) << " ";
		}
		os << "]" << endl;
	}
	return os;
}

void arrangeMatrix(Matrix& trg) {
	int i, j, row, col, num;
	row = trg.getNRows(); col = trg.getNColumns();
	num = 0;
	for (i = 0; i < row; ++i)
		for (j = 0; j < col; ++j)
			trg(i, j) = ++num;
}