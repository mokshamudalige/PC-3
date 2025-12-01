/*
 * Gaussian Blur - OpenMP Parallel Implementation
 * 
 * This implementation parallelizes the Gaussian blur convolution operation
 * using OpenMP's shared-memory model with parallel for directives.
 * 
 * Parallelization Strategy:
 * - Uses #pragma omp parallel for to distribute outer loop iterations across threads
 * - Collapse(2) clause to parallelize both i and j loops for better load balancing
 * - Private variables ensure thread-safe computation
 * - Static scheduling for predictable, balanced workload distribution
 * 
 * Author: P.D.M.D.M.Mudalige
 * ID: IT23372344
 * Course: SE3082 - Parallel Computing
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define WIDTH 1000
#define HEIGHT 1000

// Gaussian kernel for blur filter
float kernel[3][3] = {
    {1/16.0, 2/16.0, 1/16.0},
    {2/16.0, 4/16.0, 2/16.0},
    {1/16.0, 2/16.0, 1/16.0}
};

/*
 * Serial Gaussian Blur (for comparison)
 * 
 * Parameters:
 *   input  - Input image array [HEIGHT][WIDTH]
 *   output - Output blurred image array [HEIGHT][WIDTH]
 */
void gaussianBlurSerial(float input[HEIGHT][WIDTH], float output[HEIGHT][WIDTH]) {
    for (int i = 1; i < HEIGHT - 1; i++) {
        for (int j = 1; j < WIDTH - 1; j++) {
            float sum = 0.0;
            
            // Convolution operation
            for (int ki = -1; ki <= 1; ki++) {
                for (int kj = -1; kj <= 1; kj++) {
                    sum += input[i + ki][j + kj] * kernel[ki + 1][kj + 1];
                }
            }
            output[i][j] = sum;
        }
    }
}

/*
 * Parallel Gaussian Blur using OpenMP
 * 
 * Parameters:
 *   input  - Input image array [HEIGHT][WIDTH]
 *   output - Output blurred image array [HEIGHT][WIDTH]
 *   num_threads - Number of OpenMP threads to use
 */
void gaussianBlurParallel(float input[HEIGHT][WIDTH], float output[HEIGHT][WIDTH], int num_threads) {
    omp_set_num_threads(num_threads);
    
    // Parallel region with collapse clause for nested loop parallelization
    // Note: Loop variables i and j are automatically private, no need to declare
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i < HEIGHT - 1; i++) {
        for (int j = 1; j < WIDTH - 1; j++) {
            float sum = 0.0;
            
            // Convolution operation with SIMD optimization
            #pragma omp simd reduction(+:sum)
            for (int ki = -1; ki <= 1; ki++) {
                for (int kj = -1; kj <= 1; kj++) {
                    sum += input[i + ki][j + kj] * kernel[ki + 1][kj + 1];
                }
            }
            output[i][j] = sum;
        }
    }
}

/*
 * Initialize image with sample data
 */
void initializeImage(float image[HEIGHT][WIDTH]) {
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            if (i > HEIGHT/4 && i < 3*HEIGHT/4 && j > WIDTH/4 && j < 3*WIDTH/4) {
                image[i][j] = 100.0;  // Center region
            } else {
                image[i][j] = 10.0;   // Border region
            }
        }
    }
}

/*
 * Print a small portion of the image for verification
 */
void printImageSample(float image[HEIGHT][WIDTH], int sample_size) {
    printf("Sample output (%dx%d from center):\n", sample_size, sample_size);
    int start_i = HEIGHT/2 - sample_size/2;
    int start_j = WIDTH/2 - sample_size/2;
    
    for (int i = start_i; i < start_i + sample_size; i++) {
        for (int j = start_j; j < start_j + sample_size; j++) {
            printf("%6.2f ", image[i][j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {
    // Seed random number generator for reproducible results
    srand(time(NULL));
    
    int num_threads = 4;  // Default number of threads
    
    if (argc > 1) {
        num_threads = atoi(argv[1]);
        if (num_threads < 1) {
            printf("Invalid number of threads. Using default: 4\n");
            num_threads = 4;
        }
    }
    
    printf("=== Gaussian Blur - OpenMP Implementation ===\n");
    printf("Image Size: %dx%d\n", WIDTH, HEIGHT);
    printf("Number of Threads: %d\n", num_threads);
    printf("Maximum Available Threads: %d\n\n", omp_get_max_threads());
    
    // Allocate memory
    float (*input)[WIDTH] = malloc(HEIGHT * sizeof(*input));
    float (*output)[WIDTH] = malloc(HEIGHT * sizeof(*output));
    
    if (input == NULL || output == NULL) {
        printf("Memory allocation failed!\n");
        return 1;
    }
    
    // Initialize output to zero
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            output[i][j] = 0.0;
        }
    }
    
    // Initialize input image
    initializeImage(input);
    
    // Run serial version first for comparison
    printf("Running serial version for comparison...\n");
    double serial_start = omp_get_wtime();
    gaussianBlurSerial(input, output);
    double serial_end = omp_get_wtime();
    double serial_time = serial_end - serial_start;
    printf("Serial Execution Time: %.6f seconds\n\n", serial_time);
    
    // Reset output for parallel version
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            output[i][j] = 0.0;
        }
    }
    
    // Measure parallel execution time
    printf("Running parallel version with %d thread(s)...\n", num_threads);
    double start_time = omp_get_wtime();
    
    // Apply Gaussian blur
    gaussianBlurParallel(input, output, num_threads);
    
    double end_time = omp_get_wtime();
    double execution_time = end_time - start_time;
    
    // Results
    printf("Parallel Execution Time: %.6f seconds\n\n", execution_time);
    
    printImageSample(output, 5);
    
    // Performance metrics
    printf("\n=== Performance Metrics ===\n");
    printf("Total pixels processed: %d\n", WIDTH * HEIGHT);
    printf("Pixels per second (parallel): %.2f million\n", (WIDTH * HEIGHT) / (execution_time * 1e6));
    
    // Speedup and Efficiency
    double speedup = serial_time / execution_time;
    double efficiency = (speedup / num_threads) * 100.0;
    
    printf("\n=== Speedup Analysis ===\n");
    printf("Serial Time:     %.6f seconds\n", serial_time);
    printf("Parallel Time:   %.6f seconds\n", execution_time);
    printf("Speedup:         %.2fx\n", speedup);
    printf("Efficiency:      %.2f%%\n", efficiency);
    printf("Threads Used:    %d\n", num_threads);
    
    // Free memory
    free(input);
    free(output);
    
    return 0;
}
