#include <stdio.h>
#include <stdlib.h>

int n;
int km;
int *x,*y;
const float tr0 = 0.3;
float *graph;
float *trails;
float *gpu_trails;
float *gpu_graph;
float *gpu_chinfo;
float *gpu_scores;
float *gpu_paths;
curandState *states;
cudaEvent_t start,stop;
float timems;

void parse(char *);
void init_host();
void init_gpu();
void aco_free();

void parse(char *f) {
	FILE *fp = fopen(f,"r");
	if (!fp)
		die("errore lettura del file");
	fscanf(fp,"%d",&n);
	fscanf(fp,"%d",&km);
	x = (int *) malloc(n*sizeof(int));
	y = (int *) malloc(n*sizeof(int));
	for (int i = 0; i < n; i++) {
		fscanf("%d %d", &x[i],&y[i]);
	}
	fclose(fp);
}

void init_host() {
	graph = (float *) malloc(n*n*sizeof(float));
	trails = (float *) malloc(n*n*sizeof(float));
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			graph[i*n+j] = distance(x[i],y[i],x[j],y[j]);
			trails[i*n+j] = tr0;
		}
	}
}

#define gpumalloc(x,l) \
	{ \
		cudaError e = cudaMalloc((void**)x,l); \ 
		if (e != cudaSuccess) { \
			printf("allocazione fallita x: %s\n", cudaGetErrorString(e)); \
			exit(1); \
		} \
	}
void init_gpu() {
	gpumalloc(&gpu_trails,n*n*sizeof(float));
	gpumalloc(&gpu_graph,n*n*sizeof(float));
	gpumalloc(&gpu_chinfo,n*n*sizeof(float));
	gpumalloc(&gpu_scores,n*sizeof(float));
	gpumalloc(&gpu_paths,n*n*sizeof(int));
    gpumalloc(&states,n*n*sizeof(curandState));
}

void aco_free() {
	free(x);
	free(y);
	free(trails);
	free(graph);
	cudaFree(gpu_trails);
	cudaFree(gpu_graph);
	cudaFree(gpu_chinfo);
	cudaFree(gpu_scores);
	cudaFree(gpu_paths);
}

__global__ void computechinfo(float *graph,float *trails, float *chinfo) {
    int id = blockDim.x*blockIdx.x+threadIdx.x;
	float t1 = trails[id];
	float t2 = graph[id];
	float r = __powf(t1,params[0])*__powf(1.0/t2,params[1]);
	chinfo[id] = r;
}

_global__ void computepaths() {
    int tid = threadIdx.x;
    int n = blockDim.x;
    int id = threadIdx.x +blockDim.x*blockIdx.x;

    __shared__ int current;
    __shared__ int start;
    extern __shared__ int s[];
    int *visited = s;
    int *path = &s[n];
    
	visited[tid] = 1;
	path[tid] = -1;
	if (tid == 0) {
		start = curand(state)%n;
		current = start;
		visited[current] = 0;
	}
	__syncthreads();
	for (int i = 1; i < n; i++) {
		buf[tid] = chinfo[current*blockDim.x+threadIdx.x];
		if (!visited[tid])
			buf[tid] = 0.0f;
		buf[tid] *= curand_uniform(state);
		reduce[tid] = tid;
		if (tid+m<n)
			reduce[tid] = (buf[tid] > buf[tid+m])?tid:tid+m;
		__syncthreads();
		for (unsigned int s = m/2; s > 0; s >>=1) {
			if (tid < s)
				reduce[tid] = (buf[reduce[tid]]>buf[reduce[tid+s]])?reduce[tid]:reduce[tid+s];
			__syncthreads();
		}
		if (tid == reduce[0]) {
			visited[tid] = 0;
			path[current] = tid;
			current = tid;
		}
		__syncthreads();
	}
	if (path[tid] == -1)
		path[tid] = start;
	buf[tid] = graph[tid*n+path[tid]];
	if (tid+m<n)
		buf[tid] += buf[tid+m];
	__syncthreads();
	for (unsigned int s = m/2; s > 0; s >>=1) {
		if (tid < s)
			buff[tid] += buff[tid+s];
		__synchtreads();
	}
	paths[tid*n+blockIdx.x] = path[tid];
	if (tid == 0)
		scores[blockIdx.x] = buf[0];
}

__global__ void update_trails(float *trails, float *paths, float *scores) {
    int tid = threadIdx.x;
    int id = blockDim.x*blockIdx.x+threadIdx.x;
    int cid = threadIdx.x*blockDim.x+blcokDim.x;
    
    extern __shared__ int s[];
    float *buff = (float *)s;
    int *path = &s[blockDim.x];

	buff[tid] = trails[id];
	path[tid] = paths[cid];
	buff[tid] *= params[2];
	int i = paths[id];
	float l = 1.0/scores[tid];
	int x = tid;
	for (int i = 0; i < blockDim.x-1; i++)
		x = path[x];
	atomicAdd(&buff[i],l);
	atomicAdd(&buff[x],l);
	trails[id] = buff[tid];
}

int main(int argc,char *argv[]) {
    argc--,argv++;
    if (argc != 2)
        die("./aco [input] [numero di iterazioni]");
    parse(argv[0]);
    niters = atoi(argv[1]);
    init_host();
    init_gpu();
    
    rand_init<<<n,n>>>(states,time(0));

    for (int it = 0; it < niters; it++) {
        compute_chinfo();
        compute_paths();
        update_trails();
    }

}
