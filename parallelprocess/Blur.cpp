#include <iostream>
#include <opencv2/opencv.hpp>
#include <algorithm> 
#include <cmath>
#include <vector>
#include <omp.h>
#define M_PI  3.14159
using namespace std;
using namespace cv;
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
Mat ParallelBlurImage(Mat &input, float blur){
    vector<vector<float>> kernel = ParallelcreateGaussianKernel(5, blur);
    int halfSize = 5 / 2;
    Mat result = Mat::zeros(input.rows,input.cols,CV_8UC3);
    int rows=input.rows;
    int cols=input.cols;
    omp_set_num_threads(4);
    // Duyệt qua từng pixel trong hình ảnh parallel this, each thread will reponsible for block of input
    auto start2 = chrono::high_resolution_clock::now();
    #pragma omp parallel
    {
    int id=omp_get_thread_num();
    if(id==0)
    std::cout<<"Number of thread is "<<omp_get_num_threads();
    #pragma omp for collapse(2)
    for ( int i = 0; i < rows; ++i) 
    {
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
    }
    auto end2 = chrono::high_resolution_clock::now(); 
    chrono::duration<double> duration2 = end2 - start2;
    cout <<"Thoi gian thuc thi blur song song: "<<duration2.count()<<"s"<<endl;
    return result;
}