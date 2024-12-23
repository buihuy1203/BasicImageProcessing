#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

__device__ int clamp(int value, int low, int high) {
    return max(low, min(value, high));
}

__global__ void sharpenKernel(const uchar *input, uchar *output, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int kernel2D[3][3] = {
            {0, 1, 0},
            {1, -4, 1},
            {0, 1, 0}
        };
    if (y > 0 && y < rows - 1 && x > 0 && x < cols - 1) {
        int sum = 0;
        for (int k = -1; k <= 1; ++k) {
            for (int l = -1; l <= 1; ++l) {
                if ((y + k) >= 0 && (y + k) < rows && (x + l) >= 0 && (x + l) < cols) {
                    sum += kernel2D[k + 1][l + 1] * inputBuffer[(y + k) * cols + (x + l)];
                }
            }
        }
        resultBuffer[y * cols + x] = (uchar)clamp(sum, 0, 255);
    }
}

__global__ void applysharpenKernel(const uchar *input,const uchar *result, uchar *output, int rows, int cols, float sharp) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        for (int c = 0; c < 3; c++) {
            int idx = (y * width + x) * 3 + c;
            int grayIdx = y * width + x;
            int newValue = (float)input[idx] - sharp * (float)result[grayIdx];
            output[idx] = (uchar)clamp(newValue, 0, 255);
        }
    }
}

Mat ParallelSharpCUDA(Mat &input, float sharp_var) {
    // Image size
    int rows = input.rows;
    int cols = input.cols;
    Mat grayImage;
    cvtColor(input, grayImage, COLOR_BGR2GRAY);

    // Input and output data
    size_t dataGray = rows * cols * sizeof(uchar);
    size_t dataSize = rows * cols * 3 * sizeof(uchar);
    uchar *d_inputData, *d_output, *d_inputColor, *d_result;

    // Allocate device memory
    cudaMalloc(&d_inputData, dataGray);
    cudaMalloc(&d_output, dataSize);
    cudaMalloc(&d_inputColor, dataSize);
    cudaMalloc(&d_result, dataGray);

    // Copy input data to device
    cudaMemcpy(d_inputData, grayImage.data, dataGray, cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputColor, input.data, dataSize, cudaMemcpyHostToDevice);
    // Define block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, 
                  (rows + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    sharpenKernel<<<gridSize, blockSize>>>(d_inputData, d_result, rows, cols);
    applysharpenKernel<<<gridSize, blockSize>>>(d_inputColor, d_result, d_output, rows, cols, sharp_var);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy output data back to host
    Mat result(input.size(), CV_8UC3);
    cudaMemcpy(result.data, d_output, dataSize, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_inputData);
    cudaFree(d_output);
    cudaFree(d_result);
    cudaFree(d_inputColor);

    return result;
}
