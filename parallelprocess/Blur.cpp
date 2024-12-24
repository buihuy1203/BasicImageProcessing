#include <iostream>
#include <opencv2/opencv.hpp>
#include <algorithm> 
#include <cmath>
#include <vector>
#include <omp.h>
#include <CL/cl.hpp>
#include "readfile.cpp"

#define M_PI  3.14159
using namespace std;
using namespace cv;

static double BlurMPTime = 0;
static double BlurCLTime = 0;

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
Mat ParallelBlurImage(Mat &input, float blur, int process){
    //auto startSequence = chrono::high_resolution_clock::now(); 
    vector<vector<float>> kernel = ParallelcreateGaussianKernel(7, blur);
    int halfSize = 7 / 2;
    Mat result = Mat::zeros(input.rows,input.cols,CV_8UC3);
    int rows=input.rows;
    int cols=input.cols;
    omp_set_num_threads(process);
    auto start = chrono::high_resolution_clock::now();
    #pragma omp parallel for collapse(2) shared(input, result)
    for ( int i = 0; i < rows; ++i){
        for (int j = 0; j < cols; ++j) {
            Vec3f sum = Vec3f(0.0f, 0.0f, 0.0f);
            // Áp dụng kernel lên pixel cho từng kênh màu ko no need parallel since the size of kernel is small
            for (int kx = -halfSize; kx <= halfSize; ++kx) {
                for (int ky = -halfSize; ky <= halfSize; ++ky) {
                    int x = min(max(i + kx, 0), input.rows - 1);
                    int y = min(max(j + ky, 0), input.cols - 1);

                    Vec3b pixel = input.at<Vec3b>(x, y);//shared data input
                    float weight = kernel[kx + halfSize][ky + halfSize];

                    sum[0]+= pixel[0] * weight; // Kênh Blue
                    sum[1]+= pixel[1] * weight; // Kênh Green
                    sum[2]+= pixel[2] * weight; // Kênh Red
                }
            }
            // Gán giá trị đã làm mờ cho ảnh kết quả
            result.at<Vec3b>(i, j) = Vec3b(static_cast<uchar>(sum[0]), static_cast<uchar>(sum[1]), static_cast<uchar>(sum[2]));
        }
    }
    auto end = chrono::high_resolution_clock::now(); 
    //auto endSequence = chrono::high_resolution_clock::now(); 
    chrono::duration<double> duration = end - start;
    //cout <<"Blur Parralel Time: "<<duration2.count()<<"s"<<endl;
    //chrono::duration<double> durationSequence = endSequence - startSequence;
    //cout <<"Blur Process Time: "<<durationSequence.count()<<"s"<<endl;*/
    BlurMPTime += duration.count();
    return result;
}

Mat ParallelBlurOpenCL(Mat &input, float blur_set) {
    // KernelSize
    int kernelSize = 7;

    int rows = input.rows;
    int cols = input.cols;
    vector<vector<float>> kernel2D = ParallelcreateGaussianKernel(kernelSize, blur_set);
    vector<float> kernel1D(kernelSize*kernelSize);
    for (int i = 0; i < kernelSize; ++i)
        for (int j = 0; j < kernelSize; ++j)
            kernel1D[i * kernelSize + j] = kernel2D[i][j];
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
    string kernel_code = loadKernelSourceFile("kernel/blur.cl");
    sources.push_back({kernel_code.c_str(), kernel_code.length()});

    cl::Program program(context, sources);
    program.build({device});

    // Tạo buffer
    cl::Buffer bufferInput(context, CL_MEM_READ_ONLY, inputData.size());
    cl::Buffer bufferOutput(context, CL_MEM_WRITE_ONLY, outputData.size());
    cl::Buffer bufferKernel(context, CL_MEM_READ_ONLY, kernel1D.size()* sizeof(float));

    // Gửi dữ liệu lên thiết bị
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);
    queue.enqueueWriteBuffer(bufferInput, CL_TRUE, 0, inputData.size(), inputData.data());
    queue.enqueueWriteBuffer(bufferKernel, CL_TRUE, 0, kernel1D.size()*sizeof(float), kernel1D.data());
    // Tạo kernel
    cl::Kernel kernel(program, "gaussblur");
    kernel.setArg(0, bufferInput);
    kernel.setArg(1, bufferOutput);
    kernel.setArg(2, bufferKernel);
    kernel.setArg(3, rows);
    kernel.setArg(4, cols);
    kernel.setArg(5, input.channels());
    kernel.setArg(6, kernelSize);

    // Thiết lập kích thước công việc
    cl::NDRange global(rows, cols);

    auto start = std::chrono::high_resolution_clock::now();
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);
    queue.finish();
    auto end = std::chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start; // Thời gian chạy kernel (ms)
    BlurCLTime += duration.count();
    // Lấy kết quả từ thiết bị
    queue.enqueueReadBuffer(bufferOutput, CL_TRUE, 0, outputData.size(), outputData.data());
    
    // Chuyển đổi dữ liệu thành Mat
    Mat result(input.size(), CV_8UC3); // Đảm bảo khớp định dạng
    memcpy(result.data, outputData.data(), outputData.size());
    return result;
}
extern double BlurMPTime;
extern double BlurCLTime;