#include <iostream>
#include <opencv2/opencv.hpp>
#include <algorithm> 
#include <cmath>
#include <omp.h>
#include <CL/cl.hpp>
#include "readfile.hpp"
#include <vector>
using namespace std;
using namespace cv;

static double SharpMPTime = 0;
static double SharpCLTime = 0;

Mat ParallelSharpness(const Mat &input, double sharp, int process){
    //auto startSequence = chrono::high_resolution_clock::now(); 
    int procs_num= process;
    omp_set_num_threads(procs_num);
    Mat result = Mat::zeros(input.size(), input.type());
    Mat grayImage;
    cvtColor(input, grayImage, COLOR_BGR2GRAY);
    int kernel[3][3] = {
        {0, 1, 0},
        {1, -4, 1},
        {0, 1, 0}
    };
    Mat sharpenedImage = Mat::zeros(input.size(), input.type());
    auto start = chrono::high_resolution_clock::now(); 
    #pragma omp parallel for collapse(2) shared(grayImage, result)
        for (int i = 1; i < grayImage.rows - 1; ++i) {
            for (int j = 1; j < grayImage.cols - 1; ++j) {
            // Tính giá trị mới cho pixel (i, j)
            int sum = 0;
                for (int k = -1; k <= 1; ++k) {
                    for (int l = -1; l <= 1; ++l) {
                        sum += kernel[k + 1][l + 1] * grayImage.at<uchar>(i + k, j + l);
                    }
                }
            // Giới hạn giá trị để nằm trong khoảng [0, 255]
                result.at<uchar>(i, j) = saturate_cast<uchar>(sum);
        }
    }
    #pragma omp parallel for collapse(2) shared(input, sharpenedImage)
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            for (int c = 0; c < 3; ++c) { 
                int newValue = input.at<Vec3b>(i, j)[c] - sharp * result.at<uchar>(i, j);
                sharpenedImage.at<Vec3b>(i, j)[c] = saturate_cast<uchar>(newValue);
            }
        }
    }
    
    auto end = chrono::high_resolution_clock::now(); 
    //auto endSequence = chrono::high_resolution_clock::now(); 
    //chrono::duration<double> durationSequence = endSequence - startSequence;
    chrono::duration<double> duration = end - start;
    //cout <<"Sharpness Process Time: "<<durationSequence.count()<<"s"<<endl;
    //cout <<"Sharness Parallel Time: "<<durationParallel.count()<<"s"<<endl;
    SharpMPTime += duration.count();
    return sharpenedImage;
}

Mat ParallelSharpnessOpenCL(const Mat& input, float sharp_var) {

    // Convert input image to grayscale
    int rows = input.rows;
    int cols = input.cols;
    Mat grayImage;
    cvtColor(input, grayImage, COLOR_BGR2GRAY);

    vector<uchar> inputData(rows * cols);
    vector<uchar> inputColor(rows * cols * 3);
    vector<uchar> outputData(rows * cols * 3);
    memcpy(inputData.data(), grayImage.data, inputData.size());
    memcpy(inputColor.data(), input.data, input.total() * input.elemSize());
    // Initialize OpenCL
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform platform = platforms[0];

    vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    cl::Device device = devices[0];

    cl::Context context(device);
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

    // Load kernel source
    string kernelCode = loadKernelSourceFile("kernel/sharpness.cl");
    cl::Program::Sources sources;
    sources.push_back({kernelCode.c_str(), kernelCode.length()});

    // Build program
    cl::Program program(context, sources);
    program.build({device});

    // Create buffers
    cl::Buffer bufferInput(context, CL_MEM_READ_ONLY, grayImage.total());
    cl::Buffer resultBuffer(context, CL_MEM_READ_WRITE, grayImage.total());
    cl::Buffer sharpResultBuffer(context, CL_MEM_WRITE_ONLY, input.total() * input.elemSize());
    cl::Buffer bufferColor(context, CL_MEM_READ_ONLY, inputColor.size());

    queue.enqueueWriteBuffer(bufferColor, CL_TRUE, 0, inputColor.size(), inputColor.data());
    queue.enqueueWriteBuffer(bufferInput, CL_TRUE, 0, inputData.size(), inputData.data());
    // Set kernel arguments
    cl::Kernel sharpenKernel(program, "sharpen");
    cl::Kernel applySharpnessKernel(program, "applySharpness");

    sharpenKernel.setArg(0, bufferInput);
    sharpenKernel.setArg(1, resultBuffer);
    sharpenKernel.setArg(2, cols);
    sharpenKernel.setArg(3, rows);

    applySharpnessKernel.setArg(0, bufferColor);
    applySharpnessKernel.setArg(1, resultBuffer);
    applySharpnessKernel.setArg(2, sharpResultBuffer);
    applySharpnessKernel.setArg(3, sharp_var);
    applySharpnessKernel.setArg(4, cols);
    applySharpnessKernel.setArg(5, rows);

    // Run kernels
    cl::NDRange global(cols, rows);
    auto start = std::chrono::high_resolution_clock::now();
    queue.enqueueNDRangeKernel(sharpenKernel, cl::NullRange, global, cl::NullRange);
    queue.enqueueNDRangeKernel(applySharpnessKernel, cl::NullRange, global, cl::NullRange);
    queue.finish();
    auto end = std::chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start; // Thời gian chạy kernel (ms)
    SharpCLTime += duration.count();

    // Read back the result into outputData
    queue.enqueueReadBuffer(sharpResultBuffer, CL_TRUE, 0, outputData.size(), outputData.data());
    // Create the result Mat and copy data
    Mat result(input.size(), CV_8UC3);
    memcpy(result.data, outputData.data(), outputData.size());

    return result;
}

extern double SharpMPTime;
extern double SharpCLTime;