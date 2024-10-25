#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <vector>
#include <assert.h>

#define M_ROWS 3000    // Number of rows in the matrix M
#define M_COLS 1000    // Number of cols in the matrix M
// #define N_ROWS 60     // Number of rows in the matrix N
#define N_COLS 700     // Number of cols in the matrix N

#define THREADS_PER_BLOCK_X 32  // Define the number of threads in a block
#define THREADS_PER_BLOCK_Y 32  // Define the number of threads in a block

inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

__global__ void matrixMul(const float* M, const float* N, float* P, const int rows_M, const int cols_M, const int cols_N) {

    // Calculate the thread ID of the overall grid
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread computes one element of the result matrix
    if (row < rows_M && col < cols_N) {
        float sum = 0.0;
        // Accessing all ements of a row of M and a column of N
        for (int k = 0; k < cols_M; ++k) {
            sum += M[row * cols_M + k] * N[k * cols_N + col];
        }
        P[row * cols_N + col] = sum;
    }
}

// Function to generate a random number between 0 and 1
float random_number() {
    return (std::rand() * 1.0f / RAND_MAX);
}

// Function to print out the matrix
void print_matrix(const float* M, int rows, int cols) {
    int max_print_size = 10;
    rows = (rows < max_print_size) ? rows : max_print_size;
    cols = (cols < max_print_size) ? cols : max_print_size;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f", M[i * cols + j]);
            if (j < cols - 1) printf("\t");
        }
        printf("\n");
    }
}

int main(int argc, char** argv) {
    // Size in bytes for the ROWS x COLS matrix
    int size_M = M_ROWS * M_COLS * sizeof(float);  
    int size_N = M_COLS * N_COLS * sizeof(float);  
    int size_P = M_ROWS * N_COLS * sizeof(float);  

    // Host memory allocation
    float *h_M = (float*)malloc(size_M);
    float *h_N = (float*)malloc(size_N);
    float *h_P = (float*)malloc(size_P);

    // Initialize matrix M
    for (int i = 0; i < M_ROWS * M_COLS; i++) {
        h_M[i] = 1.0 + (float)rand()/RAND_MAX;
    }
    // Initialize matrix N
    for (int i = 0; i < M_COLS * N_COLS; i++) {
        h_N[i] = 1.0 + (float)rand()/RAND_MAX;
    } 
    
    printf("Matrix M\n");
    print_matrix(h_M, M_ROWS, M_COLS);

    printf("Matrix N\n");
    print_matrix(h_N, M_COLS, N_COLS);

/*     int M_ROWS = 40;     // Number of rows in the matrix M
    int M_COLS = 60;     // Number of cols in the matrix M
    // #define N_ROWS 60     // Number of rows in the matrix N
    int N_COLS = 50;     // Number of cols in the matrix N */

    /* srand(time(NULL));

    std::vector<float> h_M(M_ROWS * M_COLS), h_N(M_COLS * N_COLS), h_P(M_ROWS * N_COLS);
    std::generate(h_M.begin(), h_M.end(), random_number);
    std::generate(h_N.begin(), h_N.end(), random_number);
    
    printf("Matrix M\n");
    print_matrix(h_M.data(), M_ROWS, M_COLS);

    printf("Matrix N\n");
    print_matrix(h_N.data(), M_COLS, N_COLS); */

    // Device memory allocation
    float* d_M; 
    float* d_N;
    float* d_P;
    size_t matrixSize_M = M_ROWS * M_COLS * sizeof(float);
    size_t matrixSize_N = M_COLS * N_COLS * sizeof(float);
    size_t matrixSize_P = M_ROWS * N_COLS * sizeof(float);

    cudaMalloc((void**)&d_M, matrixSize_M);
    cudaMalloc((void**)&d_N, matrixSize_N);
    cudaMalloc((void**)&d_P, matrixSize_P);

    // Copy matrices M and N from host to device
    cudaMemcpy(d_M, h_M, matrixSize_M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, matrixSize_N, cudaMemcpyHostToDevice);

    /* cudaMemcpy(d_M, h_M.data(), matrixSize_M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N.data(), matrixSize_N, cudaMemcpyHostToDevice); */

    // Define block and grid sizes
    dim3 blockSize(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    dim3 gridSize(ceil(float(M_ROWS)/blockSize.y), ceil(float(N_COLS)/blockSize.x));
    // dim3 gridSize((N_COLS + blockSize.x - 1) / blockSize.x, (M_ROWS + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    matrixMul<<<gridSize, blockSize>>>(d_M, d_N, d_P, M_ROWS, M_COLS, N_COLS);

    // Copy the result matrix P from device to host
    checkCuda(cudaMemcpy(h_P, d_P, matrixSize_P, cudaMemcpyDeviceToHost));

    // Print part of the result matrix P for verification
    printf("Matrix P\n");
    print_matrix(h_P, M_ROWS, N_COLS);
    
    /* checkCuda(cudaMemcpy(h_P.data(), d_P, matrixSize_P, cudaMemcpyDeviceToHost));

    printf("Matrix P\n");
    print_matrix(h_P.data(), M_ROWS, N_COLS); */

    // Free device memory
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);

    // Free host memory
    free(h_M);
    free(h_N);
    free(h_P);    

    return 0;
}
