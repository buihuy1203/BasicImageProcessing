#include <iostream>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <cmath>
#include <algorithm>
#include <CL/cl.hpp>
#include "readfile.hpp"
using namespace std;
using namespace cv;
//fasdasdga
static double SatMPTime = 0;
static double SatCLTime = 0;

template<typename T>
T newclamp(const T& value, const T& low, const T& high) {
    return (value < low) ? low : (value > high) ? high : value;
}

Mat ParallelSaturation(const Mat &input, float set_sar, int process) {
    //auto startSequence = chrono::high_resolution_clock::now(); 
    int height = input.rows;
    int width = input.cols;
    Mat result = Mat::zeros(height, width, CV_8UC3);
    int procs_num = process;
    omp_set_num_threads(procs_num);
    auto start = chrono::high_resolution_clock::now(); 
    #pragma omp parallel for collapse(2) shared(input, result)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            Vec3b pixel = input.at<Vec3b>(y, x);
            float blue = static_cast<float>(pixel[0])/255.0;
            float green = static_cast<float>(pixel[1])/255.0;
            float red = static_cast<float>(pixel[2])/255.0;

            float max_channel = std::max({blue, green, red});
            float min_channel = std::min({blue, green, red});
            float delta = max_channel-min_channel;
            
            float light = (max_channel+min_channel)/2;
            float sar;
            if(delta == 0.0f)
                sar = 0;
            else
                sar = delta/(1-abs(2*light-1));
            float hue;
            if (delta == 0)
                hue = 0; // Undefined hue khi delta = 0
            else if(max_channel == red)
                hue = 60*static_cast<float>(fmod(((green-blue)/delta),6));
            else if(max_channel == green)
                hue = 60*((blue-green)/delta + 2);
            else if(max_channel == blue)
                hue = 60*((red-green)/delta + 4);
            if (hue < 0) hue += 360;
            float new_sar = std::max(0.0f, std::min(1.0f, set_sar));

            float chroma = (1-abs(2*light-1))*new_sar;
            float temp = chroma*(1-abs(static_cast<float>(fmod((hue/60.0),2))-1));
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
            result.at<Vec3b>(y, x) = Vec3b(\
                static_cast<uchar>(newclamp(round((res_blue + mem) * 255), 0.0f, 255.0f)),\
                static_cast<uchar>(newclamp(round((res_green + mem) * 255), 0.0f, 255.0f)),\
                static_cast<uchar>(newclamp(round((res_red + mem) * 255), 0.0f, 255.0f))\
            );
        }
    }
    
    auto end = chrono::high_resolution_clock::now(); 
    //auto endSequence = chrono::high_resolution_clock::now(); 
    //chrono::duration<double> durationSequence = endSequence - startSequence;
    chrono::duration<double> duration= end - start;
    //cout <<"Saturation Process Time: "<<durationSequence.count()<<"s"<<endl;
    //cout <<"Saturation Parallel Time: "<<durationParallel.count()<<"s"<<endl;
    SatMPTime += duration.count();
    return result;
}

Mat ParallelSaturationOpenCL(Mat &input, float set_sar) {
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
    string kernel_code = loadKernelSourceFile("kernel/saturation.cl");
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
    cl::Kernel kernel(program, "saturation");
    kernel.setArg(0, bufferInput);
    kernel.setArg(1, bufferOutput);
    kernel.setArg(2, rows);
    kernel.setArg(3, cols);
    kernel.setArg(4, set_sar);

    // Thiết lập kích thước công việc
    cl::NDRange global(rows, cols);
    auto start = std::chrono::high_resolution_clock::now();
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);
    queue.finish();
    auto end = std::chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start; // Thời gian chạy kernel (ms)
    SatCLTime += duration.count();
    // Lấy kết quả từ thiết bị
    queue.enqueueReadBuffer(bufferOutput, CL_TRUE, 0, outputData.size(), outputData.data());

    // Chuyển đổi dữ liệu thành Mat
    Mat result(input.size(), CV_8UC3); // Đảm bảo khớp định dạng
    memcpy(result.data, outputData.data(), outputData.size()); 
    return result;
}

extern double SatMPTime;
extern double SatCLTime;