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
Mat ParallelBrightNess(Mat &input, int bright, int process){
    auto startSequence = chrono::high_resolution_clock::now(); 
    Mat result =  Mat::zeros(input.rows, input.cols, CV_8UC3);
    int procs_num=process;
    omp_set_num_threads(procs_num);
    auto startParrallel = chrono::high_resolution_clock::now(); 
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
    /*
    #pragma omp parallel for collapse(2)   
    for (int y = 0; y < input.rows; y++) {
    // Truy cập trực tiếp vào bộ nhớ của input và result
    uchar* input_row = input.ptr<uchar>(y);  // Dữ liệu dòng y của input
    uchar* result_row = result.ptr<uchar>(y);  // Dữ liệu dòng y của result
    
    for (int x = 0; x < input.cols; x++) {
        // Truy xuất từng pixel trực tiếp
        uchar* input_pixel = input_row + x * 3;  // Mỗi pixel có 3 giá trị (RGB)
        uchar* result_pixel = result_row + x * 3;  // Dòng kết quả
        
        // Lấy giá trị RGB
        uchar blue = input_pixel[0];
        uchar green = input_pixel[1];
        uchar red = input_pixel[2];

        // Cộng sáng cho mỗi kênh và gán vào kết quả
        result_pixel[0] = saturate_cast<uchar>(blue + bright);
        result_pixel[1] = saturate_cast<uchar>(green + bright);
        result_pixel[2] = saturate_cast<uchar>(red + bright);   
    }
    }*/
    auto endParralel = chrono::high_resolution_clock::now(); 
    auto endSequence = chrono::high_resolution_clock::now(); 
    chrono::duration<double> durationSequence = endSequence - startSequence;
    chrono::duration<double> durationParallel = endParralel - startParrallel;
    cout <<"Brightness Process Time: "<<durationSequence.count()<<"s"<<endl;
    cout <<"Brightness Parallel Time: "<<durationParallel.count()<<"s"<<endl;
    return result;
}

Mat ParallelBrightnessOpenCL(Mat &input, int bright) {
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

    cl::Platform platform = platforms[1]; // Chọn nền tảng đầu tiên
    vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.empty()) {
        throw std::runtime_error("No OpenCL devices found on platform.");
    }

    cl::Device device = devices[0]; // Chọn thiết bị đầu tiên
    cl::Context context({device});
    cl::Program::Sources sources;

    // Đọc kernel từ file
    string kernel_code = loadKernelSourceFile("kernel/brightness.cl");
    sources.push_back({kernel_code.c_str(), kernel_code.length()});

    cl::Program program(context, sources);
    program.build({device});

    // Tạo buffer
    cl::Buffer bufferInput(context, CL_MEM_READ_ONLY, inputData.size());
    cl::Buffer bufferOutput(context, CL_MEM_WRITE_ONLY, outputData.size());

    // Gửi dữ liệu lên thiết bị
    cl::CommandQueue queue(context, device);
    queue.enqueueWriteBuffer(bufferInput, CL_TRUE, 0, inputData.size(), inputData.data());

    // Tạo kernel
    cl::Kernel kernel(program, "brightness");
    kernel.setArg(0, bufferInput);
    kernel.setArg(1, bufferOutput);
    kernel.setArg(2, rows);
    kernel.setArg(3, cols);
    kernel.setArg(4, bright);

    // Thiết lập kích thước công việc
    cl::NDRange global(rows, cols);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);

    // Lấy kết quả từ thiết bị
    queue.enqueueReadBuffer(bufferOutput, CL_TRUE, 0, outputData.size(), outputData.data());

    // Chuyển đổi dữ liệu thành Mat
    Mat result(input.size(), CV_8UC3); // Đảm bảo khớp định dạng
    memcpy(result.data, outputData.data(), outputData.size()); 
    return result;
}