#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

__global__ void satKernel(const uchar *input, uchar *output, int rows, int cols, float set_sar) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < rows && x < cols) {
        int idx = (y * cols + x) * 3;
        // Input Value
        float blue = (float)(input[idx])/255.0f;
        float green = (float)(input[idx+1])/255.0f;
        float red = (float)(input[idx+2])/255.0f;

        float max_channel = max(blue, max(green, red));
        float min_channel = min(blue, min(green, red));
        float delta = max_channel-min_channel;
            
        float light = (max_channel+min_channel)/2.0f;
        float sar;
        if(delta == 0.0f)
            sar = 0;
        else
            sar = delta/(1-fabs(2*light-1));
        float hue;
        if (delta == 0)
            hue = 0; // Undefined hue khi delta = 0
        else if(max_channel == red)
            hue = 60*(float)(fmod(((green-blue)/delta),6));
        else if(max_channel == green)
            hue = 60*((blue-green)/delta + 2);
        else if(max_channel == blue)
            hue = 60*((red-green)/delta + 4);
        if (hue < 0) hue += 360;
            float new_sar = max(0.0f, min(1.0f, set_sar));

            float chroma = (1-fabs(2*light-1))*new_sar;
            float temp = chroma*(1-fabs((float)(fmod((hue/60),2))-1));
            float mem = light-chroma/2;

            float res_red =0.0f, res_green = 0.0f, res_blue = 0.0f;
            if(hue < 60&&hue>=0){
                res_red=chroma;
                res_green=temp;
                res_blue = 0;
            }else if(hue <120 && hue >=60){
                res_red=temp;
                res_green=chroma;
                res_blue = 0;
            }else if(hue >=120 && hue <180){
                res_red=0;
                res_green=chroma;
                res_blue = temp;
            }else if(hue >= 180 && hue <240){
                res_red=0;
                res_green=temp;
                res_blue = chroma;
            }else if(hue >= 240 && hue <300){
                res_red=temp;
                res_green=0;
                res_blue = chroma;
            }else if(hue>=300 && hue <360){
                res_red=chroma;
                res_green=0;
                res_blue = temp;
            }
        output[idx]= (uchar)clamp(round((res_blue + mem) * 255), 0.0f, 255.0f);
        output[idx+1]= (uchar)clamp(round((res_green + mem) * 255), 0.0f, 255.0f);
        output[idx+2]= (uchar)clamp(round((res_red + mem) * 255), 0.0f, 255.0f);
    }
}

Mat ParallelSatCUDA(Mat &input, float set_sar) {
    // Image size
    int rows = input.rows;
    int cols = input.cols;

    // Input and output data
    size_t dataSize = rows * cols * 3 * sizeof(uchar);
    uchar *d_input, *d_output;

    // Allocate device memory
    cudaMalloc(&d_input, dataSize);
    cudaMalloc(&d_output, dataSize);

    // Copy input data to device
    cudaMemcpy(d_input, input.data, dataSize, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, 
                  (rows + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    satKernel<<<gridSize, blockSize>>>(d_input, d_output, rows, cols, set_sar);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy output data back to host
    Mat result(input.size(), CV_8UC3);
    cudaMemcpy(result.data, d_output, dataSize, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    return result;
}
