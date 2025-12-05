

#include <stdio.h>
#include <stdlib.h>

// Image dimensions - can be adjusted for different image sizes
#define WIDTH 5
#define HEIGHT 5


float kernel[3][3] = {
    {1/16.0, 2/16.0, 1/16.0},
    {2/16.0, 4/16.0, 2/16.0},
    {1/16.0, 2/16.0, 1/16.0}
};


void gaussianBlur(float input[HEIGHT][WIDTH], float output[HEIGHT][WIDTH]) {
    // Iterate over all pixels except the border (i=1 to HEIGHT-2)
    for (int i = 1; i < HEIGHT - 1; i++) {
        for (int j = 1; j < WIDTH - 1; j++) {
            float sum = 0.0;  // Accumulator for weighted pixel values
            
            // Apply 3x3 kernel convolution
            for (int ki = -1; ki <= 1; ki++) {       // Kernel rows (-1, 0, 1)
                for (int kj = -1; kj <= 1; kj++) {   // Kernel columns (-1, 0, 1)
                    // Multiply input pixel by corresponding kernel weight
                    sum += input[i + ki][j + kj] * kernel[ki + 1][kj + 1];
                }
            }
            
            // Store the blurred pixel value
            output[i][j] = sum;
        }
    }
}


int main() {
    
    float input[HEIGHT][WIDTH] = {
        {10, 10, 10, 10, 10},
        {10, 50, 50, 50, 10},
        {10, 50,100, 50, 10},
        {10, 50, 50, 50, 10},
        {10, 10, 10, 10, 10}
    };
    
    // Initialize output array with zeros
    float output[HEIGHT][WIDTH] = {0};
    
    // Apply Gaussian blur filter
    gaussianBlur(input, output);

    // Display the blurred result
    printf("Blurred Image:\n");
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            printf("%6.2f ", output[i][j]);
        }
        printf("\n");
    }
    
    return 0;
}

