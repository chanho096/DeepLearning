#include "Test_FNNMnist.h"
#include <fstream>
#include <iostream>
using std::cout;
using std::ifstream; using std::ios;
using std::cout; using std::endl;

int reverseInt(int i)
{
	unsigned char c1, c2, c3, c4;

	c1 = i & 255;
	c2 = (i >> 8) & 255;
	c3 = (i >> 16) & 255;
	c4 = (i >> 24) & 255;

	return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void getImage(ExMatrix& result, const string& path) {
	int magic_number = 0;
	int num_images = 0;
	int num_rows = 0;
	int num_cols = 0;
	int size, i, j;

	ifstream file(path, ios::binary);
	if (!(file.is_open())) {
		cout << "파일 열기 실패" << endl;
		return;
	}

	file.read((char *)&magic_number, sizeof(magic_number)); magic_number = reverseInt(magic_number);
	file.read((char *)&num_images, sizeof(num_images)); num_images = reverseInt(num_images);
	file.read((char *)&num_rows, sizeof(num_rows)); num_rows = reverseInt(num_rows);
	file.read((char *)&num_cols, sizeof(num_cols)); num_cols = reverseInt(num_cols);

	typedef unsigned char uchar;
	uchar** _data = new uchar*[num_images];
	size = num_rows * num_cols;
	for (i = 0; i < num_images; ++i) {
		_data[i] = new uchar[size];
		file.read((char *)_data[i], size);
	}

	result.initialize(size, num_images);
	for (i = 0; i < num_images; ++i) {
		for (j = 0; j < size; j++) {
			result(j, i) = (R)_data[i][j];
		}
	}

	for (i = 0; i < num_images; ++i) {
		delete [] _data[i];
		_data[i] = nullptr;
	}
	delete[] _data; _data = nullptr;

}

void getLabel(ExMatrix& result, const string& path, const int& labelSize) {
	int magic_number = 0;
	int num_labels = 0;
	int i, j;

	ifstream file(path, ios::binary);
	if (!(file.is_open())) {
		cout << "파일 열기 실패" << endl;
		return;
	}

	file.read((char *)&magic_number, sizeof(magic_number)); magic_number = reverseInt(magic_number);
	file.read((char *)&num_labels, sizeof(num_labels)); num_labels = reverseInt(num_labels);

	typedef unsigned char uchar;
	uchar* _data = new uchar[num_labels];
	for (i = 0; i < num_labels; ++i) {
		file.read((char *)&_data[i], 1);
	}

	result.initialize(labelSize, num_labels);
	for (i = 0; i < num_labels; ++i) {
		for (j = 0; j < labelSize; j++) {
			result(j, i) = (R)0.0;
		}
		result((int)_data[i], i) = (R)1.0;
	}

	delete[] _data; _data = nullptr;
}