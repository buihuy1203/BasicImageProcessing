#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

__global__ void blurKernel(const uchar *input, uchar *output,const float *kernel1D  int rows, int cols, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < rows && x < cols) {
        int idx = (y * cols + x) * 3;
        output[idx] = min(max(input[idx] + bright, 0), 255);         // Blue
        output[idx + 1] = min(max(input[idx + 1] + bright, 0), 255); // Green
        output[idx + 2] = min(max(input[idx + 2] + bright, 0), 255); // Red
    }
}

vector<vector<float>> ParallelcreateGaussianKernel(int size, float sigma) {
    vector<vector<float>> kernel(size, vector<float>(size));
    float sum = 0.0f;
    int halfSize = size / 2;
    for (int x = -halfSize; x <= halfSize; ++x) {
        for (int y = -halfSize; y <= halfSize; ++y) {
                kernel[x + halfSize][y + halfSize] = (1/(2*M_PI*sigma*sigma))*exp(-(x * x + y * y) / (2 * sigma * sigma));
                sum += kernel[x + halfSize][y + halfSize];
            }
        }
    // Chuẩn hóa kernel
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            kernel[i][j] /= sum;
        }
    }
    return kernel;
}

Mat ParallelBlurCUDA(Mat &input, int bright) {
    // Image size
    int rows = input.rows;
    int cols = input.cols;

    // Gaussian kernel
    int kernelSize = 7;
    vector<vector<float>> kernel2D = ParallelcreateGaussianKernel(kernelSize, blur_set);
    vector<float> kernel1D(kernelSize*kernelSize);
    for (int i = 0; i < kernelSize; ++i)
        for (int j = 0; j < kernelSize; ++j)
            kernel1D[i * kernelSize + j] = kernel2D[i][j];

    // Input and output data
    size_t dataSize = rows * cols * 3 * sizeof(uchar);
    uchar *d_input, *d_output;
    float *d_kernel;

    // Allocate device memory
    cudaMalloc(&d_input, dataSize);
    cudaMalloc(&d_output, dataSize);
    cudaMalloc(&d_kernel, 7 * 7 * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input.data, dataSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel1D.data(), 7 * 7 * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, 
                  (rows + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    blurKernel<<<gridSize, blockSize>>>(d_input, d_output,d_kernel, rows, cols, kernelSize);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy output data back to host
    Mat result(input.size(), CV_8UC3);
    cudaMemcpy(result.data, d_output, dataSize, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);

    return result;
}
