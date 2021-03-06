
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

struct bitmap {
	int n, m;
	unsigned int size;
	unsigned int *bits;
	bitmap(int rows, int cols) {
		int bitcols = (cols + 31) / 32;
		n = rows;
		m = bitcols;
		size = n*m;
		bits = new unsigned int[size];
		memset(bits, 0, size*sizeof(unsigned int));
		
	}
	~bitmap() {
		delete[] bits;
	}
	void set_bit(int row, int col) {
		bits[row*m + col / 32] |= (((unsigned int)1) << (31 - col % 32));
		
	}
	bool get_bit(int row, int col) {
		int i = row*m + col / 32;
		unsigned int flag = 1;
		flag = flag << (31 - col % 32);
		if ((flag&bits[i]) == 0)
			return false;
		else
			return true;

	}
};






//matrix[i][[j] = matrix[i*M+j]


void parse_transactions(string &file,bitmap &t) {
	ifstream csv(file, ios::in);
	string line;
	int i = 0;
	while (getline(csv, line)) {
		stringstream ss(line);
		int value;
		while (ss >> value) {
			t.set_bit(i, value);
			if (ss.peek() == ',')
				ss.ignore();
		}
		i++;
	}
	csv.close();
}

void copy_transactions_to_device(bitmap &t, unsigned int **dev_matrix) {
	cudaError_t status;
	unsigned int *temp;
	status = cudaMalloc((void **)&temp, t.n * t.m * sizeof(unsigned int));
	if (status != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_matrix");
		return;
	}
	status = cudaMemcpy(temp, t.bits, t.n * t.m * sizeof(unsigned int), cudaMemcpyHostToDevice);
	if (status != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! dev_matrix");
		return;
	}
	*dev_matrix = temp;
	
	
}


__device__ __host__  unsigned int popcount(unsigned int value) {
	unsigned int count = 0;
	while (value > 0) {           // until all bits are zero
		if ((value & 1) == 1)     // check lower bit
			count++;
		value >>= 1;              // shift bits, removing lower bit
	}
	return count;
}
__global__ void count1itemset(unsigned int *transactions, int N, int M, int *counter) {
	
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	for (int i = idx; i < N; i += blockDim.x*gridDim.x) {
		for (int j = 0; j < M; j++) {
			counter[i] += popcount(transactions[i*M + j]);
		}
	}
}	




void get_1itemset(unsigned int *dev_transactions,  int nitems, int M, int ntrans, float min_sup, bitmap &first_itemset, int &x) {
	int *counter = new int[nitems];
	int *dev_counter;
	cudaError_t status;
	status = cudaMalloc((void **)&dev_counter, nitems*sizeof(int));
	if (status != cudaSuccess) {

		fprintf(stderr, "cudamalloc failed! dev_counter %s\n", cudaGetErrorString(status));
		goto error;
	}
	status = cudaMemset(dev_counter, 0, nitems*sizeof(int));
	if (status != cudaSuccess) {

		fprintf(stderr, "cudamemset failed! dev_counter %s\n", cudaGetErrorString(status));
		goto error;
	}
	count1itemset <<<5,15>>> (dev_transactions, nitems, M, dev_counter);
	status = cudaDeviceSynchronize();
	if (status != cudaSuccess) {

		fprintf(stderr, "cudadevicesynch failed! %s\n", cudaGetErrorString(status));
		goto error;
	}
	status = cudaMemcpy(counter, dev_counter, nitems*sizeof(int), cudaMemcpyDeviceToHost);
	if (status != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! counter");
		goto error;
	}
	for (int i = 0; i < nitems; i++) {
		if (((float)counter[i]) / ntrans >= 0.05) {
			first_itemset.set_bit(0, i);
			x++;
			cout << i << " ";
		}
	}
	cout << endl;
	error:
	cudaFree(dev_counter);
	delete[] counter;
}


__global__ void gen_candidates_and_count(unsigned int* dev_transactions, unsigned int* bm, int n, unsigned int *result) {
	extern __shared__ unsigned int next[];
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	int count = 0;
	int nn = (n * (n - 1)) / 2;
	for (int i = idx; i < n; i += blockDim.x*gridDim.x) {
		for (int j = i + 1; j < n; j++) {
			for (int k = 0; k < n; k++) {
				next[(i+count)*n+k] = bm[i*n+k] | bm[j*n+k];
				count++;
			}
		}
	}
	__syncthreads();
	memcpy(result, next, nn*sizeof(unsigned int));
}
void gen_candidates(unsigned int* d_t, unsigned int *bm, int n) {
	unsigned int *dev_bm;
	cudaMalloc((void **)dev_bm, n*sizeof(unsigned int));
	cudaMemcpy(dev_bm, bm, n*sizeof(unsigned int), cudaMemcpyHostToDevice);
	int dim = ((n * (n - 1)) / 2)*n;
	unsigned int*d_result;
	cudaMalloc((void**)&d_result, dim*sizeof(unsigned int));
	cudaMemset(d_result, 0, dim*sizeof(unsigned int));
	gen_candidates_and_count<<<1,n,dim>>>(d_t, dev_bm, n, d_result);
	unsigned int *result = new unsigned int[dim];
	cudaMemcpy(result,d_result, dim*sizeof(unsigned int), cudaMemcpyDeviceToHost);

}
int main() {
	float min_sup = 0.01;
	ifstream input("input.txt", ios::in);
	string s;
	int n, ntrans, first_count = 0;
	input >> s >> n >> ntrans;
	cout << s << " " << n << " " << ntrans << endl;
	input.close();
	ntrans++;
	bitmap transactions(n, ntrans);
	bitmap first_itemset(1, n);
	parse_transactions(s, transactions);
	unsigned int *dev_transactions;
	copy_transactions_to_device(transactions, &dev_transactions);
	get_1itemset(dev_transactions, transactions.n, transactions.m, ntrans, min_sup, first_itemset, first_count);
	/*bitmap f(first_count, first_count);
	for (int i = 0; i < first_count; i++)
		f.set_bit(i, i);*/
	unsigned int *f = new unsigned int[first_count*first_count]();
	for (int i = 0; i < first_count; i++) {
		f[i*first_count+i] = 1;
	}
	gen_candidates(dev_transactions, f, first_count);
	cout << first_count << endl;
	cudaFree(dev_transactions);
}

