#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>

int *x, *y, n, m, km, blocks = 40;
float *graph, *trails, *gpu_graph = NULL, *gpu_trails = NULL,*g_buffer = NULL;
cudaEvent_t start, stop;
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

__global__ void aco(float *ggraph, float *gtrails, float *gbuffer, curandState *states, int n, int nstadi, int niters) {
    float *lgraph, *ltrails, *probs, *temp,score;
    int *path,tid,nants,current;
    extern __shared__ int smem[];
	__shared__ int fscore; //punteggio del formicaio
    curandState state;

    tid = threadIdx.x;
    nants = blockDim.x;
    state = states[blockIdx.x*blockDim.x+threadIdx.x];
    lgraph = (float *) smem;
    ltrails = (float *) (smem + n*n);
	temp = (float *) (smem +2*n*n);
    probs = (float *) (smem + 2*n*n + tid*n);
    path = (smem + 2*n*n + nants*n + tid*n);
	reduce = (sem + 2*n*n + 2*nants*n);
	
    for (int i = 0; i < n; i++) {
        for (int j = tid; j < n; j+=nants) {
            lgraph[i*n+j] = ggraph[i*n+j];
        }
    }
    __syncthreads();
    for (int i = 0; i < n; i++) {
        for (int j = tid; j < n; j+=nants) {
            ltrails[i*n+j] = gtrails[i*n+j];
        }
    }
    __syncthreads();
	for (int st = 0; st < nstadi; st++) {
		for (int it = 0; i < niters; it++) {
			score = 0.0;
		    path[0] = curand(&state)%n;
		    current = path[0];
		    for (int i = 1; i < n; i++) {
		        for (int j = 0; j < n; j++)
		            probs[j] = __powf(ltrails[current*n+j],params[0])*__powf(1.0/lgraph[current*n+j],params[1]);
		        for (int j = 0; j < i; j++)
		            probs[path[j]] = 0.0f;
		        float acc = 0;
		        for (int j = 0; j < n; j++)
		            acc += probs[j];
		        for (int j = 0; j < n; j++)
		            probs[j] /= acc;
		        for (int j = 1; j < n; j++)
		            probs[j] += probs[j-1];
		        float choice = curand_uniform(&state);
		        int k = 0;
		        for (; choice > probs[k]; k++);
		        current = k;
		        path[i] = k;
		        score += lgraph[current*n+path[i-1]];
		    }
		    score += lgraph[path[0]*n+path[n-1]];
		    for (int i = 0; i < n; i++) {
		        for (int j = tid; j< n; j+=nants) {
		            ltrails[i*n+j] *= params[2];
		        }
		    }
		    __syncthreads();
		    for (int i = 1; i < n; i++) {
		        atomicAdd(&ltrails[path[i]*n+path[i-1]], 1.0/score);
		        atomicAdd(&ltrails[path[i-1]*n+path[i]],1.0/score);
		    }
		    atomicAdd(&ltrails[path[0]*n+path[n-1]],1.0/score);
		    atomicAdd(&ltrails[path[n-1]*n+path[0]],1.0/score);
		    __syncthreads();
		}
		/*
			reduce intrablocco
		*/
		temp[tid] = score;
		for (unsigned int s = nants/2; s > 0; s >>= 1) {
			if (tid < s)
				reduce[tid] = (temp[tid] < temp[tid+s])?tid:tid+s;
			__synchthreads();
		}
		if (tid == reduce[0]) {
			fscore = score;
			gbuffer[blockIdx.x] = score;
		}
		__threadfence();
		/*
			reduce interblocco
		*/
		if (tid < gridDim.x)
			temp[tid] = gbuffer[tid];
		for (unsigned int s = gridDim.x/2; s > 0; s >>= 1) {
			if (tid < s)
				reduce[tid] = (temp[tid]  < temp[tid+s])?tid:tid+s;
			__synchthreads();
		}
		if (blockIdx.x == reduce[0]) {
			for (int i = 0; i < n; i++) {
				for (int j = tid; j < n; j += nants) {
					gtrails[i*n+j] = ltrails[i*n+j];
				}
			}
		}
		__threadfence();
		for (int i = 0; i <n; i++) {
			for (int j = tid; j < n; j += nants) {
				ltrails[i*n+j] = gtrails[i*n+j];
			}
		}
		__synchthreads();
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
    	matrice probabilità nformiche*n (float)
    	matrice percorsi nformiche*n (int)
    	buffer n int
    */
    //size_t sharedsize = 2*n*n*sizeof(float) + nformiche*n*sizeof(float) + nformiche*n*sizeof(int) + n*sizeof(float); //aggiungere reduce buffer
    size_t sharedsize = 2*n*n*sizeof(float) + nformiche*n*sizeof(float) + nformiche*n*sizeof(int) + nformiche*sizeof(int);
    printf("totale memoria shared: %zu * %d = %zu\n", sharedsize, nformicai, sharedsize*nformicai);
    /*

    */
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    rand_init<<<nformicai,nformiche>>>(states,time(0));
    cudaEventRecord(start);
    /*

    */
    aco<<<nformicai,nformiche,sharedsize>>>(gpu_graph,gpu_trails,g_buffer,states,n,nstadi,niters);

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
    //output risultati
    clean_memory();
    exit(0);
}
