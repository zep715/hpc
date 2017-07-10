
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <cassert>
using namespace std;

typedef unsigned char byte;

__global__ void count1itemset(byte *, int, int, int *);
struct Matrix {
	int N;
	int M;
	byte *matrix;
};

typedef struct Matrix Matrix;

void init_matrix(Matrix *m, int _N, int _M) {
	m->N = _N;
	m->M = _M;
	m->matrix = new byte[_N*_M]();
}

void delete_matrix(Matrix *m) {
	delete[] m->matrix;
	m->matrix = NULL;
}
void parse_metadata(char *file, string &csv, int &N, int &M) {
	ifstream metadata(file, ios::in);
	metadata >> csv >> N >> M;
	metadata.close();
}
//matrix[i][[j] = matrix[i*M+j]
void parse_line(string &line, byte *transaction, int row, int cols) {
	assert(transaction != NULL);
	stringstream ss(line);
	unsigned int value;
	while (ss >> value) {
		//transaction[value] = 1;
		transaction[value] = 1;
		if (ss.peek() == ',')
			ss.ignore();
	}

}

void parse_transactions(string &file, Matrix *m, int N, int M) {
	ifstream csv(file, ios::in);
	string line;
	int i = 0;
	while (getline(csv, line)) {
		parse_line(line, &(m->matrix[i*M]), i, M);
		i++;
	}
	csv.close();
}

void copy_transactions_to_device(Matrix *m, byte **dev_matrix) {
	cudaError_t status;
	byte *temp;
	status = cudaMalloc((void **)&temp, m->M * m->N * sizeof(byte));
	if (status != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_matrix");
		return;
	}
	status = cudaMemcpy(temp, m->matrix, m->M * m->N * sizeof(byte), cudaMemcpyHostToDevice);
	if (status != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_matrix");
		return;
	}
	*dev_matrix = temp;
	
	
}



__global__ void count1itemset(byte *transactions, int N, int M, int *counter) {
	
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++) {
			if (transactions[i*M + j]) {
				counter[j]++;
			}
		}
	}
}	



void get_1itemset(byte *dev_transactions,  int N, int M, float min_sup) {
	int *counter = new int[M];
	int *dev_counter;
	cudaMalloc((void **)&dev_counter, M*sizeof(int));
	cudaMemset(dev_counter, 0, M*sizeof(int));
	count1itemset <<<1,1>>> (dev_transactions, N, M, dev_counter);
	cudaDeviceSynchronize();
	cudaMemcpy(counter, dev_counter, M*sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < 2; i++)
		cout << counter[i] << " ";
	cout << endl;
}

__global__ void prova_gpu(byte *transactions, int N, int M, byte *res) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++) {
			res[i*M + j] = transactions[i*M + j];
		}
	}
}
void prova(Matrix *m, byte *dev_transactions, int N, int M) {
	byte *dev_counter;
	cudaError_t status;
	status = cudaMalloc((void **)&dev_counter, N*M*sizeof(byte));
	if (status != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_matrix");
		return;
	}
	prova_gpu<<<1,1>>> (dev_transactions, N, M, dev_counter);
	byte *prova = new byte[N*M];
	status = cudaMemcpy(prova, dev_counter, N*M*sizeof(byte), cudaMemcpyDeviceToHost);
	if (status != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_matrix");
		return;
	}
	cout << "----------" << endl;
	for (int i = 0; i < 20; i++) {
		for (int j = 0; j <40; j++)
			cout << (int)prova[i*M + j] << " ";
		cout << endl;
	}
	delete[] prova;
	cudaFree(dev_counter);
}
int main()
{
	int N, M;
	string csv;
	float min_sup = 0.1;
	float minconf = 0.5;

	parse_metadata("input.txt", csv, N, M);
	M++;
	Matrix matrix;
	init_matrix(&matrix, N, M);
	parse_transactions(csv, &matrix,N,M);
	for (int i = 0; i < 20; i++) {
		for (int j = 0; j <40; j++)
			cout << (int)matrix.matrix[i*M + j] << " "; 
			cout << endl;
	}
	byte *dev_transactions;
	cudaSetDevice(0);
	
	copy_transactions_to_device(&matrix, &dev_transactions);
	//get_1itemset(dev_transactions, N, M, min_sup);
	prova(&matrix, dev_transactions,  N,  M);

	cudaFree(dev_transactions);
	delete_matrix(&matrix);
	return 0;
}

