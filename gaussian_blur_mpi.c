/*
 * Gaussian Blur - MPI Parallel Implementation
 * 
 * This implementation parallelizes the Gaussian blur convolution operation
 * using MPI's distributed-memory model with domain decomposition.
 * 
 * Parallelization Strategy:
 * - Domain decomposition: image is divided into horizontal strips (rows)
 * - Each process handles a subset of rows
 * - Halo/ghost rows are exchanged between neighboring processes
 * - Master process (rank 0) distributes data and collects results
 * 
 * Author: P.D.M.D.M.Mudalige
 * ID: IT23372344
 * Course: SE3082 - Parallel Computing
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>

#define WIDTH 1000
#define HEIGHT 1000

// Gaussian kernel for blur filter
float kernel[3][3] = {
    {1/16.0, 2/16.0, 1/16.0},
    {2/16.0, 4/16.0, 2/16.0},
    {1/16.0, 2/16.0, 1/16.0}
};

/*
 * Apply Gaussian blur to a local portion of the image
 * 
 * Parameters:
 *   local_input  - Local input data including halo rows
 *   local_output - Local output data (without halo rows)
 *   local_rows   - Number of rows this process is responsible for
 */
void gaussianBlurLocal(float **local_input, float **local_output, int local_rows) {
    // Process local rows (skip halo rows at index 0 and local_rows+1)
    for (int i = 1; i <= local_rows; i++) {
        for (int j = 1; j < WIDTH - 1; j++) {
            float sum = 0.0;
            
            // Convolution operation
            for (int ki = -1; ki <= 1; ki++) {
                for (int kj = -1; kj <= 1; kj++) {
                    sum += local_input[i + ki][j + kj] * kernel[ki + 1][kj + 1];
                }
            }
            local_output[i-1][j] = sum;
        }
    }
}

/*
 * Initialize image with sample data
 */
void initializeImage(float **image) {
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
void printImageSample(float **image, int sample_size) {
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
    int rank, size;
    double start_time, end_time;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Calculate rows per process
    int rows_per_process = HEIGHT / size;
    int remainder = HEIGHT % size;
    
    // Allocate full image only on master process
    float **input = NULL;
    float **output = NULL;
    
    if (rank == 0) {
        printf("=== Gaussian Blur - MPI Implementation ===\n");
        printf("Image Size: %dx%d\n", WIDTH, HEIGHT);
        printf("Number of Processes: %d\n", size);
        printf("Rows per process: %d\n\n", rows_per_process);
        
        // Allocate full image
        input = (float **)malloc(HEIGHT * sizeof(float *));
        output = (float **)malloc(HEIGHT * sizeof(float *));
        for (int i = 0; i < HEIGHT; i++) {
            input[i] = (float *)malloc(WIDTH * sizeof(float));
            output[i] = (float *)malloc(WIDTH * sizeof(float));
        }
        
        // Initialize input image
        initializeImage(input);
        
        // Initialize output to zero
        for (int i = 0; i < HEIGHT; i++) {
            for (int j = 0; j < WIDTH; j++) {
                output[i][j] = 0.0;
            }
        }
    }
    
    // Calculate local rows for this process
    int local_rows = rows_per_process;
    if (rank < remainder) local_rows++;
    
    // Allocate local buffers (include halo rows)
    float **local_input = (float **)malloc((local_rows + 2) * sizeof(float *));
    float **local_output = (float **)malloc(local_rows * sizeof(float *));
    
    for (int i = 0; i < local_rows + 2; i++) {
        local_input[i] = (float *)malloc(WIDTH * sizeof(float));
    }
    for (int i = 0; i < local_rows; i++) {
        local_output[i] = (float *)malloc(WIDTH * sizeof(float));
    }
    
    // Start timing
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    
    // Master distributes data
    if (rank == 0) {
        int offset = 0;
        for (int p = 0; p < size; p++) {
            int send_rows = rows_per_process;
            if (p < remainder) send_rows++;
            
            if (p == 0) {
                // Copy data for master process
                for (int i = 0; i < send_rows; i++) {
                    memcpy(local_input[i+1], input[i], WIDTH * sizeof(float));
                }
                // Top halo (boundary condition - copy first row)
                memcpy(local_input[0], input[0], WIDTH * sizeof(float));
                // Bottom halo
                if (send_rows < HEIGHT) {
                    memcpy(local_input[send_rows+1], input[send_rows], WIDTH * sizeof(float));
                } else {
                    memcpy(local_input[send_rows+1], input[send_rows-1], WIDTH * sizeof(float));
                }
            } else {
                // Send to other processes
                for (int i = 0; i < send_rows; i++) {
                    MPI_Send(input[offset + i], WIDTH, MPI_FLOAT, p, i, MPI_COMM_WORLD);
                }
                // Send halo rows
                if (offset > 0) {
                    MPI_Send(input[offset - 1], WIDTH, MPI_FLOAT, p, send_rows, MPI_COMM_WORLD);
                } else {
                    MPI_Send(input[0], WIDTH, MPI_FLOAT, p, send_rows, MPI_COMM_WORLD);
                }
                
                if (offset + send_rows < HEIGHT) {
                    MPI_Send(input[offset + send_rows], WIDTH, MPI_FLOAT, p, send_rows + 1, MPI_COMM_WORLD);
                } else {
                    MPI_Send(input[HEIGHT - 1], WIDTH, MPI_FLOAT, p, send_rows + 1, MPI_COMM_WORLD);
                }
            }
            offset += send_rows;
        }
    } else {
        // Receive data from master
        for (int i = 0; i < local_rows; i++) {
            MPI_Recv(local_input[i+1], WIDTH, MPI_FLOAT, 0, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        // Receive halo rows
        MPI_Recv(local_input[0], WIDTH, MPI_FLOAT, 0, local_rows, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(local_input[local_rows+1], WIDTH, MPI_FLOAT, 0, local_rows + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    // Apply Gaussian blur locally
    gaussianBlurLocal(local_input, local_output, local_rows);
    
    // Gather results to master
    if (rank == 0) {
        // Copy master's results
        for (int i = 0; i < local_rows; i++) {
            memcpy(output[i], local_output[i], WIDTH * sizeof(float));
        }
        
        // Receive from other processes
        int offset = local_rows;
        for (int p = 1; p < size; p++) {
            int recv_rows = rows_per_process;
            if (p < remainder) recv_rows++;
            
            for (int i = 0; i < recv_rows; i++) {
                MPI_Recv(output[offset + i], WIDTH, MPI_FLOAT, p, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            offset += recv_rows;
        }
    } else {
        // Send results to master
        for (int i = 0; i < local_rows; i++) {
            MPI_Send(local_output[i], WIDTH, MPI_FLOAT, 0, i, MPI_COMM_WORLD);
        }
    }
    
    // End timing
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    
    // Master prints results
    if (rank == 0) {
        double execution_time = end_time - start_time;
        printf("Execution Time: %.6f seconds\n\n", execution_time);
        
        printImageSample(output, 5);
        
        printf("\n=== Performance Metrics ===\n");
        printf("Total pixels processed: %d\n", WIDTH * HEIGHT);
        printf("Pixels per second: %.2f million\n", (WIDTH * HEIGHT) / (execution_time * 1e6));
        
        // Free full image
        for (int i = 0; i < HEIGHT; i++) {
            free(input[i]);
            free(output[i]);
        }
        free(input);
        free(output);
    }
    
    // Free local buffers
    for (int i = 0; i < local_rows + 2; i++) {
        free(local_input[i]);
    }
    for (int i = 0; i < local_rows; i++) {
        free(local_output[i]);
    }
    free(local_input);
    free(local_output);
    
    MPI_Finalize();
    return 0;
}
