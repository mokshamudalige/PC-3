/*
 * Gaussian Blur - CUDA GPU Implementation
 * 
 * This implementation parallelizes the Gaussian blur convolution operation
 * using CUDA for GPU-based parallelization.
 * 
 * Parallelization Strategy:
 * - Each thread processes one output pixel
 * - 2D thread blocks (16x16) for optimal memory coalescing
 * - Grid dimensions calculated to cover entire image
 * - Shared memory optimization for kernel coefficients
 * - Boundary checking to handle edge cases
 * 
 * Compilation for Google Colab (Tesla T4 - Compute Capability 7.5):
 * nvcc -arch=sm_75 gaussian_blur_cuda.cu -o gaussian_blur_cuda
 * 
 * For other GPUs, use appropriate compute capability:
 * - Tesla K80: -arch=sm_37
 * - Tesla P100: -arch=sm_60
 * - Tesla V100: -arch=sm_70
 * - Tesla T4: -arch=sm_75
 * - A100: -arch=sm_80
 * 
 * Author: P.D.M.D.M.Mudalige
 * ID: IT23372344
 * Course: SE3082 - Parallel Computing
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define WIDTH 1000
#define HEIGHT 1000
#define MAX_BLOCK_SIZE 32
#define KERNEL_SIZE 3

// Gaussian kernel in constant memory for fast access
__constant__ float d_kernel[KERNEL_SIZE][KERNEL_SIZE];

/*
 * CUDA Kernel for Gaussian Blur
 * 
 * Each thread computes one output pixel by applying the convolution
 * operation to its corresponding neighborhood in the input image.
 * 
 * Parameters:
 *   input  - Input image (device memory)
 *   output - Output blurred image (device memory)
 *   width  - Image width
 *   height - Image height
 */
__global__ void gaussianBlurKernel(float *input, float *output, int width, int height) {
    // Calculate global thread indices
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Boundary check - skip border pixels and out-of-bounds threads
    if (row >= 1 && row < height - 1 && col >= 1 && col < width - 1) {
        float sum = 0.0f;
        
        // Apply convolution
        for (int ki = -1; ki <= 1; ki++) {
            for (int kj = -1; kj <= 1; kj++) {
                int input_row = row + ki;
                int input_col = col + kj;
                int input_idx = input_row * width + input_col;
                
                sum += input[input_idx] * d_kernel[ki + 1][kj + 1];
            }
        }
        
        // Write result
        int output_idx = row * width + col;
        output[output_idx] = sum;
    }
}

/*
 * CUDA Kernel for Gaussian Blur with Shared Memory Optimization
 * 
 * This version uses shared memory to cache input data for each block,
 * reducing global memory accesses and improving performance.
 */
__global__ void gaussianBlurKernelShared(float *input, float *output, int width, int height) {
    // Shared memory for input tile (includes halo region)
    __shared__ float shared_input[MAX_BLOCK_SIZE + 2][MAX_BLOCK_SIZE + 2];
    
    // Calculate global thread indices
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Calculate local thread indices
    int local_col = threadIdx.x + 1;
    int local_row = threadIdx.y + 1;
    
    // Load data into shared memory
    if (row < height && col < width) {
        shared_input[local_row][local_col] = input[row * width + col];
        
        // Load halo regions (boundary pixels)
        if (threadIdx.x == 0 && col > 0) {
            shared_input[local_row][0] = input[row * width + (col - 1)];
        }
        if (threadIdx.x == blockDim.x - 1 && col < width - 1) {
            shared_input[local_row][local_col + 1] = input[row * width + (col + 1)];
        }
        if (threadIdx.y == 0 && row > 0) {
            shared_input[0][local_col] = input[(row - 1) * width + col];
        }
        if (threadIdx.y == blockDim.y - 1 && row < height - 1) {
            shared_input[local_row + 1][local_col] = input[(row + 1) * width + col];
        }
    }
    
    __syncthreads();
    
    // Compute convolution
    if (row >= 1 && row < height - 1 && col >= 1 && col < width - 1) {
        float sum = 0.0f;
        
        for (int ki = -1; ki <= 1; ki++) {
            for (int kj = -1; kj <= 1; kj++) {
                sum += shared_input[local_row + ki][local_col + kj] * d_kernel[ki + 1][kj + 1];
            }
        }
        
        output[row * width + col] = sum;
    }
}

/*
 * Check CUDA errors
 */
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

/*
 * Initialize image with sample data
 */
void initializeImage(float *image) {
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            int idx = i * WIDTH + j;
            if (i > HEIGHT/4 && i < 3*HEIGHT/4 && j > WIDTH/4 && j < 3*WIDTH/4) {
                image[idx] = 100.0f;  // Center region
            } else {
                image[idx] = 10.0f;   // Border region
            }
        }
    }
}

/*
 * Print a small portion of the image for verification
 */
void printImageSample(float *image, int sample_size) {
    printf("Sample output (%dx%d from center):\n", sample_size, sample_size);
    int start_i = HEIGHT/2 - sample_size/2;
    int start_j = WIDTH/2 - sample_size/2;
    
    for (int i = start_i; i < start_i + sample_size; i++) {
        for (int j = start_j; j < start_j + sample_size; j++) {
            printf("%6.2f ", image[i * WIDTH + j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {
    int block_size = 16;  // Default block size
    int use_shared_memory = 0;  // Default: use basic kernel
    
    if (argc > 1) {
        block_size = atoi(argv[1]);
        if (block_size < 1 || block_size > MAX_BLOCK_SIZE) {
            printf("Invalid block size. Using default: 16\n");
            block_size = 16;
        }
    }
    if (argc > 2) {
        use_shared_memory = atoi(argv[2]);
    }
    
    printf("=== Gaussian Blur - CUDA Implementation ===\n");
    printf("Image Size: %dx%d\n", WIDTH, HEIGHT);
    printf("Block Size: %dx%d\n", block_size, block_size);
    printf("Kernel Version: %s\n\n", use_shared_memory ? "Shared Memory" : "Basic");
    
    // Device properties
    cudaDeviceProp prop;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, 0));
    printf("GPU Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Max Threads per Block: %d\n\n", prop.maxThreadsPerBlock);
    
    // Host memory allocation
    size_t image_size = WIDTH * HEIGHT * sizeof(float);
    float *h_input = (float *)malloc(image_size);
    float *h_output = (float *)malloc(image_size);
    
    if (h_input == NULL || h_output == NULL) {
        printf("Host memory allocation failed!\n");
        return 1;
    }
    
    // Initialize input image
    initializeImage(h_input);
    
    // Initialize output to zero
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        h_output[i] = 0.0f;
    }
    
    // Device memory allocation
    float *d_input, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, image_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, image_size));
    
    // Copy kernel to constant memory
    float h_kernel[KERNEL_SIZE][KERNEL_SIZE] = {
        {1/16.0f, 2/16.0f, 1/16.0f},
        {2/16.0f, 4/16.0f, 2/16.0f},
        {1/16.0f, 2/16.0f, 1/16.0f}
    };
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_kernel, h_kernel, 
                     KERNEL_SIZE * KERNEL_SIZE * sizeof(float)));
    
    // Copy input data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, image_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_output, h_output, image_size, cudaMemcpyHostToDevice));
    
    // Configure kernel launch parameters
    dim3 block_dim(block_size, block_size);
    dim3 grid_dim((WIDTH + block_size - 1) / block_size, 
                  (HEIGHT + block_size - 1) / block_size);
    
    printf("Grid Dimensions: %dx%d blocks\n", grid_dim.x, grid_dim.y);
    printf("Total Threads: %d\n\n", grid_dim.x * grid_dim.y * block_size * block_size);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    
    // Launch kernel and measure time
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    
    if (use_shared_memory) {
        gaussianBlurKernelShared<<<grid_dim, block_dim>>>(d_input, d_output, WIDTH, HEIGHT);
    } else {
        gaussianBlurKernel<<<grid_dim, block_dim>>>(d_input, d_output, WIDTH, HEIGHT);
    }
    
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    // Check for kernel launch errors
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Calculate execution time
    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    float execution_time = milliseconds / 1000.0f;
    
    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, image_size, cudaMemcpyDeviceToHost));
    
    // Results
    printf("Execution Time: %.6f seconds\n\n", execution_time);
    
    printImageSample(h_output, 5);
    
    // Performance metrics
    printf("\n=== Performance Metrics ===\n");
    printf("Total pixels processed: %d\n", WIDTH * HEIGHT);
    printf("Pixels per second: %.2f million\n", (WIDTH * HEIGHT) / (execution_time * 1e6));
    printf("GPU Throughput: %.2f GFLOPS\n", 
           (WIDTH * HEIGHT * 9 * 2) / (execution_time * 1e9));  // 9 mult + 9 add per pixel
    
    // Cleanup
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    free(h_input);
    free(h_output);
    
    return 0;
}
