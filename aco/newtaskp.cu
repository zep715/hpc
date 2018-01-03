#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>

int *x, *y, n, m, km, blocks = 40;
float *graph, *trails, *gpu_graph = NULL, *gpu_trails = NULL,*g_buffer = NULL;
float hparams[3] = {1.0, 5.0, 0.5};
__constant__ float params[3];

void parse(char *f) {
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

__global__ void aco(float *gpu_graph, float *gpu_trails, float *g_buffer, curandState *states,int nnodi,int nstadi, int iters_per_stadio) {
	int tid = threadIdx.x;
    int id = threadIdx.x + blockDim.x*blockIdx.x;
    int n = blockDim.x;
    curandState state = states[id];
	__shared__ float punteggio;
    extern __shared__ int s[];
	int size = nnodi*nnodi;
	/*
	ripartizione shared memory
	*/
	float *lgraph = (float *) s;
	float *ltrails = (float *) &s[size];
	float *probs = (float *) &s[2*size + tid*nnodi];
	int *path =  &s[2*size+n*nnodi+tid*n];
	float *buffer = (float *) &s[2*size+2*n*nnodi];
	//copia grafo e tracce da globale a shared
    for (int i = 0; i < n; i++)
        lgraph[i*n+tid] = gpu_graph[i*n+tid];
    for (int i = 0; i < n; i++)
        ltrails[i*n+tid] = gpu_trails[i*n+tid];
    __syncthreads();
	float length = 0.0;
	for (int stadio = 0; stadio < nstadi; stadio++) {
		for (int iters = 0; iters < iters_per_stadio; iters++) {
			path[0] = curand(&state)%n;
			int current = path[0];
			for (int i = 1; i < n; i++) {
				for (int j = 0; j < n; j++)
					probs[j] = __powf(ltrails[current*n+j],params[0])*__powf(1.0/lgraph[current*n+j],params[1]);
				for (int j = 0; j < i; j++)
					probs[path[j]] = 0.0;
				float acc = 0.0;
				for (int j = 0; j < n; j++)
					acc += probs[j];
				for (int j = 0; j < n; j++)
					probs[j] /= acc;
				for (int j = 1; j < n; j++)
					probs[j] = probs[j-1];
				float choice = curand_uniform(&state);
				int k = 0;
				while (choice > probs[k])
					k++;
				current = k;
				path[i] = k;
				length += lgraph[path[i]*n+path[i-1]];
			}
			length += lgraph[path[0]*n+path[n-1]];
			for (int i = 0; i < n; i++)
				ltrails[i*n+tid] *= params[2];
			for (int i = 1; i <n; i++) {
				atomicAdd(&ltrails[path[i]*n+path[i-1]], 1.0/length);
            	atomicAdd(&ltrails[path[i-1]*n+path[i]], 1.0/length);
			}
			atomicAdd(&ltrails[path[0]*n+path[n-1]], 1.0/length);
			atomicAdd(&ltrails[path[n-1]*n+path[0]], 1.0/length);
			__syncthreads();
		}
		
		/*
		reduce intrablocco
		*/
		buffer[tid] = length;
		int *temp_buff = &s[3*n*n];
		for (unsigned int s = n/2; s > 0; s>>=1) {
		    if (tid<s)
		        temp_buff[tid] = (buffer[tid] < buffer[tid+s]) ? tid:tid+s;
		    __syncthreads();
		}
		if (tid == temp_buff[0]) {
		    g_buffer[blockIdx.x] = length;
			punteggio = length;
		}
		__threadfence();
		/*
		reduce interblocco
		*/
		for (int i = 0; i < n; i++) {
			if (punteggio == g_buffer[i]) {
				for (int j = 0; j < n; j++)
					gpu_trails[i*n+tid] = ltrails[i*n+tid];
			} 
		}
		__threadfence();
		for (int i = 0; i < n; i++)
			ltrails[i*n+tid] = gpu_trails[i*n+tid];
		__syncthreads();
			
	}

}

int main(int argc, char *argv[]) {
	argc--,argv++;
	if (argc != 5) {
		printf("./acotaskp [file] [numero di formicai] [numero di formiche per formicaio] [numero stadi] [numero iterazioni per stadio]\n");
		exit(1);
	}
	int nformicai = atoi(argv[1]);
	int nformiche = atoi(argv[2]); //formiche per formicaio
	int nstadi = atoi(argv[3]);
	int niters_per_stadio = atoi(argv[4]);
	/*

	*/
	parse(argv[0]);
	graph = init_graph(n,x,y);
    trails = init_trails(n,0.3);
    gpu_graph = (float *) gpucopy(graph,n*n*sizeof(float));
	cudaEvent_t start, stop;
    if (!gpu_graph)
        abort("errore nella copia del grafo alla gpu\n");
    gpu_trails = (float *) gpucopy(trails,n*n*sizeof(float));
    if (!gpu_trails)
        abort("errore nella copia della matrice delle tracce alla gpu\n");
    if (cudaMemcpyToSymbol(params,hparams,3*sizeof(float)) != cudaSuccess)
        abort("errore nella copia dei parametri alla gpu\n");
	curandState *states;
    if (cudaMalloc(&states,nformicai*nformiche*sizeof(curandState)) != cudaSuccess)
        abort("errore nell'inizializzazione dell'rng su gpu\n");
	if(cudaMalloc(&g_buffer,nformicai*sizeof(float)) != cudaSuccess)
        abort("errore allocazione buffer globale nella gpu\n");
	/*
		calcolo shared memory necessaria
		matrice grafo + tracce 2*n*n (float)
		matrice probabilità nformiche * n (float)
		matrice percorsi nformiche *n (int)
		buffer n float
	*/
	size_t sharedsize = 2*n*n*sizeof(float) + nformiche*n*sizeof(float) + nformiche*n*sizeof(int) + n*sizeof(float); //aggiungere reduce buffer
	printf("total shared size: %zu * %d = %zu\n", sharedsize, nformicai, sharedsize*nformicai);
	/*

	*/
	cudaEventCreate(&start);
    cudaEventCreate(&stop);
    rand_init<<<blocks,n>>>(states,time(0));
    cudaEventRecord(start);
	/*

	*/
	aco<<<nformicai,nformiche,sharedsize>>>(gpu_graph,gpu_trails,g_buffer,states,n,nstadi,niters_per_stadio);
	
	cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms,start,stop);
    fprintf(stderr,"%f\n",ms);
	cudaError err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		printf("errore dopo lancio kernel: %s\n", cudaGetErrorString(err));
		clean_memory();
		exit(1);
	}
	//fare output risultati
	clean_memory();
	exit(0);
}
