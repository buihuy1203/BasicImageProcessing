#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

__device__ int clamp(int value, int low, int high) {
    return max(low, min(value, high));
}

__global__ void blurKernel(const uchar *input, uchar *output,const float *kernel1D  int rows, int cols, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int halfSize = kernelSize/2;

    if (x < rows && y < cols) {
        float sum[3] = {0.0f, 0.0f, 0.0f};

        for (int kx = -halfSize; kx <= halfSize; ++kx) {
            for (int ky = -halfSize; ky <= halfSize; ++ky) {
                int px = clamp(x + kx, 0, rows - 1);
                int py = clamp(y + ky, 0, cols - 1);

                int pixelIndex = (px * cols + py) * channels;
                float weight = kernel1D[(kx + halfSize) * kernelSize + (ky + halfSize)];

                sum[0] += (float)input[pixelIndex] * weight;       // Blue channel
                sum[1] += (float)input[pixelIndex + 1] * weight;   // Green channel
                sum[2] += (float)input[pixelIndex + 2] * weight;   // Red channel
            }
        }

        int outputIndex = (x * cols + y) * channels;
        output[outputIndex] = (uchar)clamp(sum[0], 0.0f, 255.0f);
        output[outputIndex + 1] = (uchar)clamp(sum[1], 0.0f, 255.0f);
        output[outputIndex + 2] = (uchar)clamp(sum[2], 0.0f, 255.0f);
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
