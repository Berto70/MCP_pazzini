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

// Function to generate a random number between 0 and 99
float random_number() {
    return (std::rand()*1./RAND_MAX);
}

int main() {
    // Seed the random number generator with the current time
    srand(time(NULL));  // Ensure that rand() produces different sequences each run

    // Local vectors hosted in memory, each with N elements
    // using a vector to host the matrix, in a row-wise allocation
    std::vector<float> h_A(ROWS * COLS), h_B(ROWS * COLS), h_C(ROWS * COLS);
    std::generate(h_A.begin(), h_A.end(), random_number);  // Fill vector 'A' with random number
    std::generate(h_B.begin(), h_B.end(), random_number);  // Fill vector 'B' with random number

    // Size in bytes for the ROWS x COLS matrix
    int size = ROWS * COLS * sizeof(float);  

    float *d_A, *d_B, *d_C;

    // Device memory allocation
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy matrices A and B from host to device
    cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    int N_tpb = 256;
    int N_blocks = ceil(float(ROWS)*COLS/N_tpb);

    // Launch the kernel
    matrixAdd<<<N_blocks, N_tpb>>>(d_A, d_B, d_C, ROWS, COLS);

    // Copy the result matrix C from device to host
    cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost);

    // Print part of the result matrix C for verification
    printf("Matrix C\n");
    print_matrix(h_C.data(), 10, 10);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    // free(C.data());

    return 0;
}
