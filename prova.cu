
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
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

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
		transaction[row*cols + value] = 1;
		if (ss.peek() == ',')
			ss.ignore();
	}

}

void parse_transactions(string &file, byte *transactions, int N, int M) {
	ifstream csv(file, ios::in);
	string line;
	int i = 0;
	while (getline(csv, line)) {
		parse_line(line, transactions, i, M);
		i++;
	}
	csv.close();
}
__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void count1itemset(byte *transactions, int *N, int *M, int *counter) {
	int element = threadIdx.x;
	for (int i = 0; i < *N; i++) {
		for (int j = 0; j < *M; j++) {
			if (transactions[i*(*M)+j] == element)
				counter[element]++;
		}
	}
}

cudaError_t get_1itemset(byte *transactions, int N, int M, float min_sup) {
	int *counter, *dev_N, *dev_M;
	int *ret = new int[M];
	byte *t;
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto error;
	}
	cudaStatus = cudaMalloc((void**)&counter, M*sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! counter");
		goto error;
	}
	cudaStatus = cudaMalloc((void**)&t, N*M*sizeof(byte));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! transactions");
		goto error;
	}
	
	cudaStatus = cudaMemcpy(t, transactions, N * M * sizeof(byte), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! transactions");
		goto error;
	}
	cudaMalloc((void **)&dev_N, sizeof(int));
	cudaMalloc((void **)&dev_M, sizeof(int));
	cudaMemcpy(dev_N, &N, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_M, &M, sizeof(int), cudaMemcpyHostToDevice);
	cout << "before calling" << endl;
	count1itemset<<<1, M >>>(t, dev_N, dev_M, counter); //count?
	cout << "after calling" << endl;
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto error;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto error;
	}
	cudaStatus = cudaMemcpy(ret, counter, M * sizeof(int), cudaMemcpyDeviceToHost);
	float temp;
	for (int i = 0; i < M; i++) {
		temp = ((float)counter[i]) / N;
		if (temp > min_sup)
			cout << i << endl;
	}
	error:
	delete[] ret;
	cudaFree(counter);
	cudaFree(t);
	return cudaStatus;
}
int main()
{
	/*
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
	*/
	int N, M;
	string csv;
	float min_sup = 0.1;
	float minconf = 0.5;

	parse_metadata("input.txt", csv, N, M);
	M++;
	byte *transactions = new byte[N*M]();
	//int **transactions = new int*[N];
	
	parse_transactions(csv, transactions,N,M);
	get_1itemset(transactions, N, M, min_sup);

	delete[] transactions;
	return 0;
}



//suddivisione matrice?


// Helper function for using CUDA to add vectors in parallel.
/*
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
*/
