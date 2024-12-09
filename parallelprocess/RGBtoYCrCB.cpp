#include <iostream>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <cmath>
#include <algorithm>
using namespace std;
using namespace cv;

Mat ParallelYCrCBImage(const Mat &input, int process) {
    auto startSequence = chrono::high_resolution_clock::now(); 
    int height = input.rows;
    int width = input.cols;
    Mat result = Mat::zeros(height, width, CV_8UC3);
    int procs_num=process;
    omp_set_num_threads(procs_num);
    auto startParrallel = chrono::high_resolution_clock::now(); 
    #pragma omp parallel for collapse(2) shared(input, result)
     for (int y = 0; y < height; y++) {
         for (int x = 0; x < width; x++) {
             Vec3b pixel = input.at<Vec3b>(y, x);
             uchar blue = pixel[0];
             uchar green = pixel[1];
             uchar red = pixel[2];

            uchar Y = static_cast<uchar>(0.299 * red + 0.587 * green + 0.114 * blue);
            uchar Cb = static_cast<uchar>(128 + (blue - Y) * 0.564);
            uchar Cr = static_cast<uchar>(128 + (red - Y) * 0.713);

            result.at<Vec3b>(y, x) = Vec3b(Y, Cb, Cr);
         }
     }


    // Parallelize the outer loop and inner loop with SIMD and collapse
    /*auto startParrallel = chrono::high_resolution_clock::now(); 
    #pragma omp parallel for collapse(2) shared(input, result)
    for (int y = 0; y < height; y++) {
        // Truy cập trực tiếp vào bộ nhớ của input và result
         const uchar* input_row = input.ptr<uchar>(y);  // Dữ liệu dòng y của input
         uchar* result_row = result.ptr<uchar>(y);  // Dữ liệu dòng y của result

        // SIMD processing
        for (int x = 0; x < width; x++) {
         const   uchar* input_pixel = input_row + x * 3;  // Mỗi pixel có 3 giá trị (RGB)
            uchar* result_pixel = result_row + x * 3;  // Dòng kết quả
            
            // Truy xuất giá trị RGB
            uchar blue = input_pixel[0];
            uchar green = input_pixel[1];
            uchar red = input_pixel[2];

            // Tính toán YCbCr
            uchar Y = static_cast<uchar>(0.299 * red + 0.587 * green + 0.114 * blue);
            uchar Cb = static_cast<uchar>(128 + (blue - Y) * 0.564);
            uchar Cr = static_cast<uchar>(128 + (red - Y) * 0.713);

            // Gán kết quả
            result_pixel[0] = Y;
            result_pixel[1] = Cb;
            result_pixel[2] = Cr;
        }
    }*/
    auto endParralel = chrono::high_resolution_clock::now(); 
    auto endSequence = chrono::high_resolution_clock::now(); 
    chrono::duration<double> durationSequence = endSequence - startSequence;
    chrono::duration<double> durationParallel = endParralel - startParrallel;
    cout <<"YCrCB Process Time: "<<durationSequence.count()<<"s"<<endl;
    cout <<"YCrCB Parallel Time: "<<durationParallel.count()<<"s"<<endl;
    return result;  
}