#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>
#include "utils.h"

#define TRAILS_INIT 0.3f

int n;
int km;
int *x,*y;
int nformicai;
int nformiche;
int nstadi;
int niters;
unsigned int m;
float *graph;
float *trails;
float *gpu_graph;
float *gpu_trails;
float *gpu_scores = NULL;
float *gpu_slots = NULL;
curandState *gpu_states = NULL;
float hparams[3] = {1.0,5.0,0.5};
__constant__ float params[3];

void parse(char*);
float distance(int,int,int,int);
void init_host(void);
void init_gpu(void);
void aco_free(void);

void parse(char *f) {
    FILE *fp = fopen(f,"r");
    if (!fp)
        die("errore nell'apertura del file di input");
    fscanf(fp,"%d",&n);
    fscanf(fp,"%d",&km);
    x = (int *) malloc(n*sizeof(int));
    y = (int *) malloc(n*sizeof(int));
    for (int i = 0; i < n; i++)
        fscanf(fp,"%d %d", &x[i],&y[i]);
    fclose(fp);
}

float distance(int x1,int y1, int x2, int y2) {
    int x=x2-x1;
    int y=y2-y1;
    return sqrt(x*x+y*y);
}

void init_host() {
    graph = (float*) malloc(n*n*sizeof(float));
    if (!graph)
        die("host_graph");
    trails = (float*) malloc(n*n*sizeof(float));
    if (!trails)
        die("host_trails");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            graph[i*n+j] = distance(x[i],y[i],x[j],y[j]);
            trails[i*n+j] = TRAILS_INIT;
        }
    }
    unsigned int temp = (unsigned int) nformiche;
    m = 1;
    while (temp >>= 1) m<<=1;

}

void init_gpu() {
    gpumalloc(&gpu_graph,n*n*sizeof(float));
    gpumalloc(&gpu_trails,n*n*sizeof(float));
    gpumalloc(&gpu_scores,nformicai*nformiche*sizeof(float));
    gpumalloc(&gpu_states,nformicai*nformiche*sizeof(curandState));
    gpumalloc(&gpu_slots,nformicai*n*n*sizeof(float));
    if(cudaMemcpyToSymbol(params,hparams,3*sizeof(float)) != cudaSuccess)
        die("parametri");
    gpucopy(gpu_graph,graph,n*n*sizeof(float));
    gpucopy(gpu_trails,trails,n*n*sizeof(float));
}

__global__ void rand_init(curandState *states, long seed) {
    int id = threadIdx.x+blockDim.x*blockIdx.x;
    curand_init(seed,id,0,&states[id]);
}

__global__ void aco_stadio(
    int n,
    int niters,
    unsigned int m,
    float *ggraph,
    float *gtrails,
    float *gscores,
    float *gdest,
    curandState *states
) {
    int tid = threadIdx.x;
    int nants = blockDim.x;
    curandState *state = &states[blockIdx.x*blockDim.x+threadIdx.x];
    float *dest = &gdest[blockIdx.x*n*n];
    float score;
    int current;

    extern __shared__ int s[];
    float *lgraph = (float *) s;
    float *ltrails = (float *) &s[n*n];
    int *path = &s[2*n*n+tid*n];
    float *probs = (float *) &s[2*n*n+nants*n+tid*n];

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
    for (int it = 0; it < niters; it++) {
        score = 0.0;
        path[0] = curand(state)%n;
        current = path[0];
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < n; j++) {
                probs[j] = __powf(ltrails[current*n+j],params[0]);
                probs[j] *= __powf(1.0/lgraph[current*n+j],params[1]);
            }
            for (int j = 0; j < n; j++)
                probs[path[j]] = 0.0f;
            float acc = 0.0f;
            for (int j = 0; j < n; j++)
                acc += probs[j];
            for (int j = 0; j < n; j++)
                probs[j] /= acc;
            for (int j = 1; j < n; j++)
                probs[j] += probs[j-1];
            float choice = curand_uniform(state);
            int k = 0;
            for (; choice > probs[k]; k++);
            current = k;
            path[i] = k;
            score += lgraph[current*n+path[i-1]];
        }
        score += lgraph[path[0]*n+path[n-1]];
        for (int i = 0; i < n; i++) {
            for (int j = tid; j < n; j+= nants) {
                ltrails[i*n+j] *= params[2];
            }
        }
        __syncthreads();

        for (int i = 1; i < n; i++) {
            atomicAdd(&ltrails[path[i]*n+path[i-1]],1.0/score);
            atomicAdd(&ltrails[path[i-1]*n+path[i]],1.0/score);
        }
        atomicAdd(&ltrails[path[0]*n+path[n-1]],1.0/score);
        atomicAdd(&ltrails[path[n-1]*n+path[0]],1.0/score);
    }

    for (int i = 0; i < n; i++) {
        for (int j = tid; j < n; j+=nants) {
            dest[i*n+j] = ltrails[i*n+j];
        }
    }
    __syncthreads();
    gscores[blockIdx.x*blockDim.x+threadIdx.x] = score;
    for (int i = 0; i < n; i++) {
        for (int j = tid; j<n; j+=nants) {
            dest[i*n+j] = ltrails[i*n+j];
        }
    }
    __syncthreads();
}
// blocchi: 1, threads: nformiche*formicai
//shared: 2*nformicai*nformiche
__global__ void selectbest(int nformicai, int nformiche, int trailsDim,int m,float *gscores, float *gslots, float *gtrails) {
    __shared__ int block;
    extern __shared__ int s[];
    int *reduce = s;
    float *buf = (float *) &s[blockDim.x];
    int tid = threadIdx.x;
    buf[tid] = gscores[tid];
    reduce[tid] = tid;
    if (tid + m < blockDim.x) {
        reduce[tid] = (buf[tid] < buf[tid+m])?tid:tid+m ;
    }
    __syncthreads();
    for (unsigned int s = m/2; s > 0; s >>= 1) {
        if (tid<s)
            reduce[tid] = (buf[reduce[tid]] < buf[reduce[tid+s]])?reduce[tid]:reduce[tid+s];
        syncthreads();
    }
    if (tid == reduce[0]) {
        block = tid/nformiche;
    }
    __syncthreads();
    float *src = &gslots[block*trailsDim*trailsDim];
    for (int i = tid; i < trailsDim*trailsDim; i+=blockDim.x)
        gtrails[i] = src[i];
}
void aco_free() {
    free(x);
    free(y);
    free(graph);
    free(trails);
    if (gpu_graph) cudaFree(gpu_graph);
    if (gpu_trails) cudaFree(gpu_trails);
    if (gpu_scores) cudaFree(gpu_scores);
    if (gpu_states) cudaFree(gpu_states);
    if (gpu_slots) cudaFree(gpu_slots);
}
int main(int argc, char *argv[]) {
    argc--,argv++;
    if (argc != 5)
        die("./aco [input] [#formicai] [#formiche] [#stadi] [#iteritazioni]");
    parse(argv[0]);
    nformicai = atoi(argv[1]);
    nformiche = atoi(argv[2]);
    nstadi = atoi(argv[3]);
    niters = atoi(argv[4]);
    init_host();
    init_gpu();
    printf("%d nodi\n",n);
    printf("%d formicai\n",nformicai);
    printf("%d formiche per formicaio\n",nformiche);
    printf("%d stadi\n",nstadi);
    printf("%d iterazioni per stadio\n",niters);
    printf("m: %d\n", m);
    atexit(aco_free);
    size_t sharedsize = n*n*sizeof(float) + n*n*sizeof(float) + nformiche*n*sizeof(int) + nformiche*n*sizeof(float);
    size_t sharedsize2 = nformicai*nformiche*sizeof(float) + nformicai*nformiche*sizeof(int);
    rand_init<<<nformicai,nformiche>>>(gpu_states,time(0));
    int m2 =1;
    int temp2 = nformicai*nformiche;
    while (temp2>>=1) m2<<=1;
    for (int st = 0; st < nstadi; st++) {
        aco_stadio<<<nformicai,nformiche,sharedsize>>>(n,niters,m,gpu_graph,gpu_trails,gpu_scores,gpu_slots,gpu_states);
        selectbest<<<1,nformicai*nformiche,sharedsize2>>>(nformicai,nformiche,n,m2,gpu_scores,gpu_slots,gpu_trails);
    }
    cudaError e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        printf("kernel fallito: %s\n", cudaGetErrorString(e));
        exit(1);
    }
    /*
    float temp[nformicai*nformiche];
    cudaMemcpy(temp,gpu_scores,nformicai*nformiche*sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < nformicai; i++) {
    	for (int j = 0; j < nformiche; j++) {
    		printf("%f ",temp[i*n+j]);
    	}
    	printf("\n");
    }
    */
    exit(0);
}
