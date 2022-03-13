#include <iostream>
#include <unistd.h>
#include <cublas_v2.h>

using namespace std;

#define CUDA_CALL(call)                                                  \
do {                                                                     \
    const cudaError_t error_code = call;                                 \
    if (error_code != cudaSuccess) {                                     \
        printf("CUDA Error:\n");                                         \
        printf("    File:       %s\n", __FILE__);                        \
        printf("    Line:       %d\n", __LINE__);                        \
        printf("    Error Code: %d\n", error_code);                      \
        printf("    Error Text: %s\n", cudaGetErrorString(error_code));  \
        exit(1);                                                         \
    }                                                                    \
} while (0)

#define KERNEL_CUBLASGEMMEX 1

typedef struct GemmArgs {
    int M, N, K;
    void *alpha, *beta, *A, *B, *C;
} GemmArgs_t;

void createSample(int M, int N, int K, GemmArgs_t **args) {
    *args = (GemmArgs_t*)malloc(sizeof(GemmArgs_t));
    (*args)->M = M;
    (*args)->N = N;
    (*args)->K = K;
    
    // ignored memory initilization

    // (*args)->alpha = malloc(sizeof(int32_t));
    // (*args)->beta  = malloc(sizeof(int32_t));
    // (*args)->A     = malloc(sizeof(int32_t)*M*K);
    // (*args)->B     = malloc(sizeof(int32_t)*K*N);
    // (*args)->C     = malloc(sizeof(int32_t)*M*N);
}

void deleteSample(GemmArgs_t *args) {
    // ignored memory initialization 
    free(args);
}

float speedTest(int kernel, GemmArgs_t *sample) {
    // prepare device data
    int M = sample->M;
    int N = sample->N;
    int K = sample->K;
    int32_t h_alpha = 1, h_beta = 1;
    void *d_alpha, *d_beta, *d_A, *d_B, *d_C;
    CUDA_CALL(cudaMalloc(&d_alpha, sizeof(int32_t)));
    CUDA_CALL(cudaMalloc(&d_beta, sizeof(int32_t)));
    CUDA_CALL(cudaMalloc(&d_A, sizeof(int32_t)*M*K));
    CUDA_CALL(cudaMalloc(&d_B, sizeof(int32_t)*K*N));
    CUDA_CALL(cudaMalloc(&d_C, sizeof(int32_t)*M*N));
    cublasHandle_t handle;
    cublasCreate(&handle);

    // ignored memory copy
    const int one = 1;
    cudaMemcpy(d_alpha, &one, sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta , &one, sizeof(int32_t), cudaMemcpyHostToDevice);

    // start timing
    CUDA_CALL(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));
    CUDA_CALL(cudaEventRecord(start));
    cudaEventQuery(start);
    CUDA_CALL(cudaDeviceSynchronize());

    //=======================  Timing Code Block  ==============================

    switch(kernel) {
        case KERNEL_CUBLASGEMMEX:
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &h_alpha, d_A, 
                         CUDA_R_8I, M, d_B, CUDA_R_8I, K, &h_beta, d_C, CUDA_R_32I,
                         M, CUDA_R_32I, CUBLAS_GEMM_DEFAULT);
            break;
        default:
            break;
    }
    
    //==========================================================================

    // end timing
    CUDA_CALL(cudaDeviceSynchronize());
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));
    float elapsed_time;
    CUDA_CALL(cudaEventElapsedTime(&elapsed_time, start, stop));
    CUDA_CALL(cudaEventDestroy(start));
    CUDA_CALL(cudaEventDestroy(stop));

    // clean up the memory
    CUDA_CALL(cudaFree(d_alpha));
    CUDA_CALL(cudaFree(d_beta));
    CUDA_CALL(cudaFree(d_A));
    CUDA_CALL(cudaFree(d_B));
    CUDA_CALL(cudaFree(d_C));

    return elapsed_time;
}

int main(int argc, char *argv[]) {
    int kernel = 0;
    int opt;
    const char *optstring = "k:";

    while ((opt = getopt(argc, argv, optstring)) != -1) {
        switch (opt) {
            case 'k':
                sscanf(optarg, "%d", &kernel);
                break;
            default:
                printf("Unknown argument %c\n", opt);
                exit(1);
        }
    }

    GemmArgs_t *args;
    for (int i = 1; i <= 30; i++) {
        int len = i << 10;
        createSample(len, len, len, &args);

        float elapsed_time = speedTest(kernel, args);
        float gflops = 2.*1e-6 * len * len * len / elapsed_time;
        printf("(%5d): %.5f ms    %f GFLOPS\n", len, elapsed_time, gflops);

        deleteSample(args);
    }
}