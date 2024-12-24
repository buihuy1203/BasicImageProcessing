#include <iostream>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <cmath>
#include <algorithm>
#include <CL/cl.hpp>
#include "readfile.hpp"
using namespace std;
using namespace cv;

static double YCrCBMPTime = 0;
static double YCrCBCLTime = 0;

Mat ParallelYCrCBImage(const Mat &input, int process) {
    //auto startSequence = chrono::high_resolution_clock::now(); 
    int height = input.rows;
    int width = input.cols;
    Mat result = Mat::zeros(height, width, CV_8UC3);
    int procs_num=process;
    omp_set_num_threads(procs_num);
    auto start = chrono::high_resolution_clock::now(); 
    #pragma omp parallel for collapse(2) shared(input, result)
     for (int y = 0; y < height; y++) {
         for (int x = 0; x < width; x++) {
             Vec3b pixel = input.at<Vec3b>(y, x);
             uchar blue = pixel[0];
             uchar green = pixel[1];
             uchar red = pixel[2];

            uchar Y = (uchar)(0.299f * red + 0.587f * green + 0.114f * blue);
            uchar Cb = (uchar)(128.0f + (blue - Y) * 0.564f);
            uchar Cr = (uchar)(128.0f + (red - Y) * 0.713f);
            result.at<Vec3b>(y, x) = Vec3b(Y, Cb, Cr);
         }
     }
    
    auto end = chrono::high_resolution_clock::now(); 
    //auto endSequence = chrono::high_resolution_clock::now(); 
    //chrono::duration<double> durationSequence = endSequence - startSequence;
    chrono::duration<double> duration = end - start;
    YCrCBMPTime += duration.count();
    //cout <<"YCrCB Process Time: "<<durationSequence.count()<<"s"<<endl;
    //cout <<"YCrCB Parallel Time: "<<durationParallel.count()<<"s"<<endl;*/
    return result;  
}

Mat ParallelYCrCBOpenCL(Mat &input) {
    // Kích thước ảnh
    int rows = input.rows;
    int cols = input.cols;

    // Tạo mảng đầu vào và đầu ra
    vector<uchar> inputData(input.rows * input.cols * 3);
    vector<uchar> outputData(input.rows * input.cols * 3);
    memcpy(inputData.data(), input.data, input.total() * input.elemSize());

    // Khởi tạo OpenCL
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        throw runtime_error("No OpenCL platforms found.");
    }

    cl::Platform platform = platforms[0]; // Chọn nền tảng đầu tiên
    vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.empty()) {
        throw std::runtime_error("No OpenCL devices found on platform.");
    }

    cl::Device device = devices[0]; // Chọn thiết bị đầu tiên
    cl::Context context({device});
    cl::Program::Sources sources;

    // Đọc kernel từ file
    string kernel_code = loadKernelSourceFile("kernel/ycrcb.cl");
    sources.push_back({kernel_code.c_str(), kernel_code.length()});

    cl::Program program(context, sources);
    program.build({device});

    // Tạo buffer
    cl::Buffer bufferInput(context, CL_MEM_READ_ONLY, inputData.size());
    cl::Buffer bufferOutput(context, CL_MEM_WRITE_ONLY, outputData.size());

    // Gửi dữ liệu lên thiết bị
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);
    queue.enqueueWriteBuffer(bufferInput, CL_TRUE, 0, inputData.size(), inputData.data());

    // Tạo kernel
    cl::Kernel kernel(program, "ycrcb");
    kernel.setArg(0, bufferInput);
    kernel.setArg(1, bufferOutput);
    kernel.setArg(2, rows);
    kernel.setArg(3, cols);

    // Thiết lập kích thước công việc
    cl::NDRange global(rows, cols);
    cl::Event event;
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);
    event.wait();
    cl_ulong startTime = 0;
    cl_ulong endTime = 0;
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &startTime);
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &endTime);
    double kernelTime = (endTime - startTime) / 1.0e9;
    YCrCBCLTime += kernelTime;
    // Lấy kết quả từ thiết bị
    queue.enqueueReadBuffer(bufferOutput, CL_TRUE, 0, outputData.size(), outputData.data());
    
    // Chuyển đổi dữ liệu thành Mat
    Mat result = Mat::zeros(rows,cols, CV_8UC3);
    memcpy(result.data, outputData.data(), outputData.size()); 
    return result;
}

extern double YCrCBMPTime;
extern double YCrCBCLTime;