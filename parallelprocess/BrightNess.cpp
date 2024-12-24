#include <iostream>
#include <opencv2/opencv.hpp>
#include <algorithm> 
#include <cmath>
#include <omp.h>
#include <chrono>
#include <string>
#include <CL/cl.hpp>
#include "readfile.hpp"
using namespace std;
using namespace cv;

static double BrightMPTime = 0;
static double BrightCLTime = 0;

Mat ParallelBrightNess(Mat &input, int bright, int process){
    //auto startSequence = chrono::high_resolution_clock::now(); 
    Mat result =  Mat::zeros(input.rows, input.cols, CV_8UC3);
    int procs_num=process;
    omp_set_num_threads(procs_num);
    auto start = chrono::high_resolution_clock::now(); 
    #pragma omp parallel for collapse(2) shared(input, result)
    for(int y=0; y < input.rows; y++){
         for(int x = 0; x < input.cols; x++){
             Vec3b pixel = input.at<Vec3b>(y,x);
             uchar blue = pixel[0];
             uchar green = pixel[1];
             uchar red = pixel[2]; 
             result.at<Vec3b>(y, x) = Vec3b(saturate_cast<uchar>(blue+bright), saturate_cast<uchar>(green+bright), saturate_cast<uchar>(red+bright));
         }
    }
    
    auto end = chrono::high_resolution_clock::now(); 
    //auto endSequence = chrono::high_resolution_clock::now(); 
    chrono::duration<double> duration = end - start;
    BrightMPTime += duration.count();
   // chrono::duration<double> durationParallel = endParralel - startParrallel;
    //cout <<"Brightness Process Time: "<<durationSequence.count()<<"s"<<endl;
    //cout <<"Brightness Parallel Time: "<<durationParallel.count()<<"s"<<endl;
    return result;
}

Mat ParallelBrightnessOpenCL(Mat &input, int bright) {
    // Image Size
    int rows = input.rows;
    int cols = input.cols;

    // Input Output Data
    vector<uchar> inputData(input.rows * input.cols * 3);
    vector<uchar> outputData(input.rows * input.cols * 3);
    memcpy(inputData.data(), input.data, input.total() * input.elemSize());

    // OpenCL Initialization
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        throw runtime_error("No OpenCL platforms found.");
    }

    cl::Platform platform = platforms[0]; 
    vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.empty()) {
        throw std::runtime_error("No OpenCL devices found on platform.");
    }

    cl::Device device = devices[0];
    cl::Context context({device});
    cl::Program::Sources sources;

    // Kernel Source
    string kernel_code = loadKernelSourceFile("kernel/brightness.cl");
    sources.push_back({kernel_code.c_str(), kernel_code.length()});

    cl::Program program(context, sources);
    program.build({device});

    // Buffer Initialization
    cl::Buffer bufferInput(context, CL_MEM_READ_ONLY, inputData.size());
    cl::Buffer bufferOutput(context, CL_MEM_WRITE_ONLY, outputData.size());

    // Send data to device
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);
    queue.enqueueWriteBuffer(bufferInput, CL_TRUE, 0, inputData.size(), inputData.data());

    // Kernel Initialization
    cl::Kernel kernel(program, "brightness");
    kernel.setArg(0, bufferInput);
    kernel.setArg(1, bufferOutput);
    kernel.setArg(2, rows);
    kernel.setArg(3, cols);
    kernel.setArg(4, bright);

    // Work Size
    cl::NDRange global(rows, cols);
    cl::Event event;
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);
    event.wait();
    cl_ulong startTime = 0;
    cl_ulong endTime = 0;
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &startTime);
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &endTime);
    double kernelTime = (endTime - startTime) / 1.0e9;
    BrightCLTime += kernelTime;
    // Read Result
    queue.enqueueReadBuffer(bufferOutput, CL_TRUE, 0, outputData.size(), outputData.data());
    // Convert Result
    Mat result(input.size(), CV_8UC3); // Đảm bảo khớp định dạng
    memcpy(result.data, outputData.data(), outputData.size());
    return result;
}

extern double BrightMPTime;
extern double BrightCLTime;