#include <stdio.h>
#include <stdlib.h>

#define WIDTH 5
#define HEIGHT 5

float kernel[3][3] = {
    {1/16.0, 2/16.0, 1/16.0},
    {2/16.0, 4/16.0, 2/16.0},
    {1/16.0, 2/16.0, 1/16.0}
};

void gaussianBlur(float input[HEIGHT][WIDTH], float output[HEIGHT][WIDTH]) {
    for (int i = 1; i < HEIGHT - 1; i++) {
        for (int j = 1; j < WIDTH - 1; j++) {
            float sum = 0.0;
            for (int ki = -1; ki <= 1; ki++) {
                for (int kj = -1; kj <= 1; kj++) {
                    sum += input[i + ki][j + kj] * kernel[ki + 1][kj + 1];
                }
            }
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
    float output[HEIGHT][WIDTH] = {0};
    gaussianBlur(input, output);

    printf("Blurred Image:\n");
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            printf("%6.2f ", output[i][j]);
        }
        printf("\n");
    }
    return 0;
}
