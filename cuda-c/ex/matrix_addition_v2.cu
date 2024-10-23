#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <vector>

#define ROWS 4000  // Number of rows in the matrices
#define COLS 6000  // Number of columns in the matrices

__global__ void matrixAdd(float* A, float* B, float* C, int rows, int cols) {

    // 1d
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread computes one element of the result matrix
    if (idx < rows * cols) {
        C[idx] = A[idx] + B[idx];
    }
}

void print_matrix(const float* A, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f", A[i * rows + j]);
            if (j < cols - 1) printf("\t");
        }
        printf("\n");
    }
}

int main() {
    
    // Size in bytes for the ROWS x COLS matrix
    int size = ROWS * COLS * sizeof(float);  

    // Host memory allocation
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize matrices A and B
    for (int i = 0; i < ROWS * COLS; i++) {
        h_A[i] = 1.0 + (float)rand()/RAND_MAX;
        h_B[i] = 2.0 + (float)rand()/RAND_MAX;
    }

    // Device memory allocation
    float *d_A, *d_B, *d_C;

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy matrices A and B from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    int N_tpb = 256;
    int N_blocks = ceil(float(ROWS)*COLS/N_tpb);

    // Launch the kernel
    matrixAdd<<<N_blocks, N_tpb>>>(d_A, d_B, d_C, ROWS, COLS);

    // Copy the result matrix C from device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print part of the result matrix C for verification
    printf("Matrix C\n");
    print_matrix(h_C, 10, 10);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    // free(C.data());

    return 0;
}
