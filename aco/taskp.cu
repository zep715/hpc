#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>

int *x, *y, n, m, km, blocks = 40;
float *graph, *trails, *gpu_graph = NULL, *gpu_trails = NULL;
float hparams[3] = {1.0, 5.0, 0.5};
__constant__ float params[3];

void parse_from_data(char *f) {
    FILE *fp= fopen(f,"r");
    fscanf(fp, "%d", &n);
    fscanf(fp, "%d", &km);
    x = (int *)malloc(n*sizeof(int));
    y = (int *)malloc(n*sizeof(int));
    for (int i = 0; i < n; i++)
        fscanf(fp, "%d %d", &x[i], &y[i]);
    fclose(fp);

}

float distance(int x1, int y1, int x2, int y2) {
    int x = x2-x1;
    int y = y2-y1;
    return sqrt(x*x+y*y);
}

float *init_graph(int n, int *x, int*y) {
    float *g = (float *) malloc(n*n*sizeof(float));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            g[i*n+j] = distance(x[i], y[i], x[j], y[j]);
        }
    }
    return g;
}

float *init_trails(int n, float v) {
    float *x = (float *) malloc(n*n*sizeof(float));
    for (int i = 0; i < n*n; i++)
        x[i] = v;
    return x;
}


void *gpucopy(void *b, size_t l) {
    void *x;
    if(cudaMalloc(&x, l) != cudaSuccess)
        return NULL;
    if(cudaMemcpy(x,b,l,cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(x);
        return NULL;
    }
    return x;
}

__device__ float build_path(float *lgraph, float *ltrails,int *own_path, float *own_probs,curandState *state) {
        int current = curand(state)%n;
        *own_path = current;
        for (int j = 1; j < n; j++) {
            for (int k=0; k < n; k++)
                own_probs[k] = __powf(ltrails[current*n+k],params[0])*__powf(1.0/lgraph[current*n+k],params[1]);
            for (int k = 0; k < j; k++)
                own_probs[own_path[k]] = 0.0;
            float acc = 0.0;
            for (int k = 0; k < n; k++)
                acc += own_probs[k];
            for (int k = 0; k < n; k++)
                own_probs[k] /= acc;
            for (int k = 1; k < n; k++)
                own_probs[k] += own_probs[k-1];
            float choice = curand_uniform(state);
            int k = 0;
            while (choice > probs[tid*n+k])
                k++;
            current = k;
            own_path[j] = k;
        }
        float length = 0.0;
        for (int j = 1; j < n; j++)
            length += lgraph[own_path[j]*n+own_path[j-1]];
        return length;
}

__global__ void aco(float *ggraph,float *gtrails,curandState *states,float *g_buffer, int niters) {
    int tid = threadIdx.x;
    int id = threadIdx.x + blockDim.x*blockIdx.x;
    int n = blockDim.x;
    curandState state = &states[id];
    extern __shared__ int s[];
    float *lgraph = (float *)s;
    float *ltrails = (float *) &s[n*n];
    float *probs = (float *) &s[2*n*n];
    int *paths = &s[3*n*n];
    int *own_path = &paths[tid*n];
    float *own_probs = &probs[tid*n];
    int *red_buffer = &s[4*n*n];
    //copia grafo e tracce da globale a shared
    for (int i = 0; i < n; i++)
        lgraph[i*n+tid] = ggraph[i*n+tid];
    for (int i = 0; i < n; i++)
        ltrails[i*n+tid] = gtrails[i*n+tid];
    __syncthreads();
    //algoritmo
    float length;
    for (int it = 0; it < niters; it++) {
        length = build_solutions(lgraph,ltrails,own_path,own_probs,state);
        for (int j = 0; j < n; j++)
            ltrails[j*n+tid] *= params[2];
        for (int j = 1; j < n; j++) {
            atomicAdd(&ltrails[own_path[j]*n+own_path[j-1]], 1.0/length);
            atomicAdd(&ltrails[own_path[j-1]*n+own_path[j]], 1.0/length);
        }
        __syncthreads();
    }
    
    //reduce intrablocco dove per determinare la formica con il percorso più breve che lo scrive in global
    probs[tid] = length;
    for (unsigned int s = blockDim.x/2; s > 0; s>>=1) {
        if (tid<s)
            paths[tid] = (probs[tid] < probs[tid+s]) ? tid:tid+s;
        __syncthreads();
    }
    if (tid == paths[0])
        g_buffer[blockIdx.x] = length;
    __threadfence();

    //reduce interblocco per determinare il blocco che ha la formica con il percorso migliore
    if (tid < gridDim.x)
        probs[tid] = g_buffer[tid];
    __syncthreads();
    for (unsigned int s = gridDim.x/2; s > 0; s >>=1) {
        if (tid < s)
            red_buffer[tid] = (probs[tid] < probs[tid+s]) ? tid : tid+s;
        __syncthreads();
    }
    //il formicaio con la miglior formica setta la matrice globale delle tracce con la propria
    if (blockIdx.x == red_buffer[0]) {
        for (int i = 0; i<n; i++)
            gtrails[i*n+tid] = ltrails[i*n+tid];
    }
    __threadfence();
    //copia di nuovo la matrice della tracce dalla memoria globale in shared
    for (int i = 0; i < n; i++)
        ltrails[i*n+tid] = gtrails[i*n+tid];
    //ripete l'algoritmo
    for (int it = 0; it < niters; it++) {
        length = build_solutions(lgraph,ltrails,own_path,own_probs,state);
        for (int j = 0; j < n; j++)
            ltrails[j*n+tid] *= params[2];
        for (int j = 1; j < n; j++) {
            atomicAdd(&ltrails[own_path[j]*n+own_path[j-1]], 1.0/length);
            atomicAdd(&ltrails[own_path[j-1]*n+own_path[j]], 1.0/length);
        }
        __syncthreads();
    }
    probs[tid] = length;
    for (unsigned int s = blockDim.x/2; s > 0; s>>=1) {
        if (tid<s)
            paths[tid] = (probs[tid] < probs[tid+s]) ? tid:tid+s;
        __syncthreads();
    }
    if (tid == paths[0])
        g_buffer[blockIdx.x] = length;
    __threadfence();
}

__global__ void rand_init(curandState *states,long seed) {
    int id = threadIdx.x+blockDim.x*blockIdx.x;
    curand_init(seed,id,0,&states[id]);
}

void clean_memory() {
    free(x);
    free(y);
    free(graph);
    free(trails);
    if (gpu_graph)
        cudaFree(gpu_graph);
    if (gpu_trails)
        cudaFree(gpu_trails);
}

void abort(const char *s) {
    printf("%s\n",s);
    clean_memory();
    exit(1);
}

int main(int argc, char* argv[]) {
    cudaEvent_t start, stop;
    size_t sharedsize;
    float *g_buffer;
    if (argc < 2) {
        printf("no input file\n");
        return 1;
    }
    if (argc == 3) {
        blocks = atoi(argv[2]);
    }
    parse_from_data(argv[1]);
    m = n;
    graph = init_graph(n,x,y);
    trails = init_trails(n,0.3);
    gpu_graph = (float *) gpucopy(graph,n*n*sizeof(float));
    if (!gpu_graph)
        abort("error in copying graph to gpu\n");
    gpu_trails = (float *) gpucopy(trails,n*n*sizeof(float));
    if (!gpu_trails)
        abort("error in copying trails to gpu\n");
    if (cudaMemcpyToSymbol(params,hparams,3*sizeof(float)) != cudaSuccess)
        abort("error in copying parameters to gpu\n");
    if(cudaMalloc(&g_buffer,blocks*sizeof(float)) != cudaSuccess)
        abort("error in allocating g_buffer to gpu\n");
    sharedsize = 3*n*n*sizeof(float) + m*n*sizeof(int) + blocks*sizeof(int);
    curandState *states;
    if (cudaMalloc(&states,blocks*n*sizeof(curandState)) != cudaSuccess)
        abort("error in allocating curand states to gpu\n");
    printf("total shared memory %zu*%d=%zu\n",sharedsize,blocks,sharedsize*blocks);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    rand_init<<<blocks,n>>>(states,time(0));
    cudaEventRecord(start);
    aco<<<blocks,m,sharedsize>>>(gpu_graph,gpu_trails,states,g_buffer,100);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms,start,stop);
    printf("%f\n",ms);
    float buffer[blocks];
    cudaDeviceSynchronize();
    cudaMemcpy(buffer,g_buffer,blocks*sizeof(float),cudaMemcpyDeviceToHost);
    for (int i = 0; i < blocks; i++)
        printf("%f ", buffer[i]);
    printf("\n");
    clean_memory();
    return 0;
}
