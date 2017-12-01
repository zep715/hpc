#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>


/*
strategia di parallelizzazione:
    prima fase:
        ogni blocco costruisce una riga della matrice choice_info
    seconda fase:
        ogni blocco costruisce un percorso in questo modo
            probs[i] = probs[i] * visited[i] * rand[i]
            next_city = reduce(probs,max)
        ogni blocco copia il proprio percorso in memoria globale e la lunghezza di tale percorso
            path[i]=x indica che dalla città i si va alla città x
    terza fase:
        ogni blocco carica in shared una riga della matrice delle tracce e la aggiorna
        applica evaporazione
        i percorsi sono in una matrice m*n dove i percorsi di ogni formica sono le colonne di questa matrice
        ogni blocco carica una riga della matrice dei percorsi e la usa per aggiornare la riga della matrice delle tracce
        affinché sia possibile aggiornare entrambi i versi (aggiornando tracce[i][j] si deve aggiornare anche tracce[j][i])
        ogni blocco carica in memoria shared un singolo percorso e sostituisce gli indici con gli elementi
        e ripete l'operazione di aggiornamento sopra descritta





*/
/*
dati globali del programma
x,y: vettori dei punti delle citta
n: numero di citta
m: numero di formiche
km:lato del quadrato che racchiude tutti i punti
graph,trails, etc: puntatori a matrici host e gpu
hparams[0] = alfa, [1] = beta, [2] = rho
params = parametri sulla gpu
*/
int *x, *y, n, m, km;
float *graph, *trails, *gpu_graph = NULL, *gpu_trails = NULL;
float *g_lengths = NULL, *g_paths = NULL,*g_choice_info = NULL;
float hparams[3] = {1.0, 5.0, 0.5};
__constant__ float params[3];

/*
fa il parsing del file di input così formato:
n
km
x1 y1
...
xn yn 
*/
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

/*
ritorna distanza euclidea tra due punti
*/
float distance(int x1, int y1, int x2, int y2) {
    int x = x2-x1;
    int y = y2-y1;
    return sqrt(x*x+y*y);
}

/*
alloca e riempie matrice delle incidenze di n*n elementi
*/
float *init_graph(int n, int *x, int*y) {
    float *g = (float *) malloc(n*n*sizeof(float));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            g[i*n+j] = distance(x[i], y[i], x[j], y[j]);
        }
    }
    return g;
}

/*
alloca e inizializza matrice delle tracce con il valore v
*/
float *init_trails(int n, float v) {
    float *x = (float *) malloc(n*n*sizeof(float));
    for (int i = 0; i < n*n; i++)
        x[i] = v;
    return x;
}

/*
alloca e copia sulla gpu il buffer b di lunghezza l
*/
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

__global__ void aco(float *ggraph, float *gtrails,float *choice_info, float *gpaths, float *glengths,curandState *states, int niters) {
    int n = blockDim.x;
    int tid = threadIdx.x;
    int id = blockDim.x*blockIdx.x+threadIdx.x;
    curandState state = states[id];
    __shared__ int current;
    __shared__ int start;
    extern __shared__ int s[];
    //ripartizione shared memory
    int *visited = s;
    float *buff = (float *)&s[n];
    int *path = &s[2*n];
    int *reduce = &s[3*n];

    for (int it = 0; it < niters; it++) {
        visited[tid] = 1;
        path[tid] = -1;
        //build choice info
        {
            float first = gtrails[id];
            float second = ggraph[id];
            first = __powf(first,params[0]);
            second = __powf(1.0/second, params[1]);
            choice_info[id] = first*second;
        }
        __threadfence();
        //build solutions
        if (tid == 0) {
            start = curand(&state)%n;
            current = start;
            visited[current] = 0;
        }
        __syncthreads();
        for (int i = 1; i < n; i++) {
            buff[tid] = choice_info[current*n+tid];
            buff[tid] *= visited[tid];
            buff[tid] *= curand_uniform(&state);
            //reduce(buff,max);
            for (unsigned int s = n/2; s > 0; s >>= 1) {
                if (tid < s) {
                    reduce[tid] = (buff[tid] > buff[tid+s]) ? tid:tid+s;
                }
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
        __syncthreads();
        //sum reduce per ottenere la lunghezza del percorso in buff[0]
        buff[tid] = ggraph[tid*n+path[tid]];
        for (unsigned int s = n/2; s > 0; s >>= 1) {
            if (tid < s)
                buff[tid] += buff[tid+s];
            __syncthreads();
        }
        //ogni blocco scrive il proprio percorso in memoria globale
        gpaths[tid*n+blockIdx.x] = path[tid];
        //un solo thread mette la lunghezza del percorso del proprio blocco in memoria globale
        if (tid == 0)
            glengths[blockIdx.x] = buff[0];
        __threadfence();
        //aggiorna le tracce--
        buff[tid] = gtrails[id];
        buff[tid] *= params[2];
        int i = gpaths[id];
        float l = 1.0/glengths[tid];
        atomicAdd(&buff[i],l);
        visited[tid] = gpaths[tid*n+blockIdx.x];
        __syncthreads();
        reduce[visited[tid]]=tid;
        __syncthreads();
        gpaths[tid*n+blockIdx.x] = reduce[tid];
        __threadfence();
        i = gpaths[id];
        atomicAdd(&buff[i],l);
        gtrails[id] = buff[tid];
        __threadfence();
    }

}
/*
kernel per l'inizializzazione delle strutture necessarie
per la generazione di numeri casuali
*/
__global__ void rand_init(curandState *states,long seed) {
    int id = threadIdx.x+blockDim.x*blockIdx.x;
    curand_init(seed,id,0,&states[id]);
}

/*
libera memoria allocata dinamicamente, host e gpu
*/
void clean_memory() {
    free(x);
    free(y);
    free(graph);
    free(trails);
    if (gpu_graph)
        cudaFree(gpu_graph);
    if (gpu_trails)
        cudaFree(gpu_trails);
    if (g_lengths)
        cudaFree(g_lengths);
    if (g_paths)
        cudaFree(g_paths);
    if (g_choice_info)
        cudaFree(g_choice_info);
}

/*
stampa messaggio di errore a video, libera la memoria
e interrompe esecuzione del programma
*/
void abort(const char *s) {
    printf("%s\n",s);
    clean_memory();
    exit(1);
}

int main(int argc, char* argv[]) {
    cudaEvent_t start, stop;
    size_t sharedsize;

    if (argc < 2) {
        printf("no input file\n");
        return 1;
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
    if(cudaMalloc(&g_lengths,n*sizeof(float)) != cudaSuccess)
        abort("error in allocating g_lengths to gpu\n");
    if(cudaMalloc(&g_paths,n*n*sizeof(int)) != cudaSuccess)
        abort("error in allocating g_paths to gpu\n");
    if(cudaMalloc(&g_choice_info,n*n*sizeof(float)) != cudaSuccess)
        abort("error in allocating choice_info to gpu\n");
    //sharedsize
    sharedsize = 3*n*sizeof(int)+n*sizeof(float);
    //
    curandState *states;
    if (cudaMalloc(&states,n*n*sizeof(curandState)) != cudaSuccess)
        abort("error in allocating curand states to gpu\n");
    printf("total shared memory %zu*%d=%zu\n",sharedsize,n,sharedsize*n);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    rand_init<<<n,n>>>(states,time(0));
    cudaEventRecord(start);
    aco<<<n,n,sharedsize>>>(gpu_graph,gpu_trails,g_choice_info,g_paths,g_lengths,states,100);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms,start,stop);
    printf("%f\n",ms);
    clean_memory();
    return 0;
}
