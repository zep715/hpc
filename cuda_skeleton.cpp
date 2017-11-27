#include <stdio.h>


__global__ void aco(ggraph, gtrails, niters) {

    for (int i = 0; i < blockDim.x; i++)
        lgraph[i*blockDim.x+threadIdx.x] = ggraph[i*blockDim+threadIdx.x];
    __syncthreads();
    for (int i = 0; i < blockDim.x; i++)
        ltrails[i*blockDim.x+threadIdx.x] = gtrails[i*blockDim+threadIdx.x];
    __syncthreads();
    for (int i = 0; i < niters; i++) {
        paths[threadIdx.x*blockDim.x] = rand() % n;
        for (int j = 1; j < n; j++) {
            paths[threadIdx*blockDim.x+j] = next;
        }
        float length = 0.0;
        for (int j = 1; j < n; j++)
            length += lgraph[path[j]*blockDim.x+path[j-1]]
        
    }
    
}
