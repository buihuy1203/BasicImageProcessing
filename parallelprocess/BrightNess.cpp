#include <iostream>
#include <opencv2/opencv.hpp>
#include <algorithm> 
#include <cmath>
#include <omp.h>
#include <chrono>
using namespace std;
using namespace cv;

Mat ParallelBrightNess(Mat &input, int bright){
    auto startSequence = chrono::high_resolution_clock::now(); 
    Mat result =  Mat::zeros(input.rows, input.cols, CV_8UC3);
     int procs_num;
     procs_num = omp_get_num_procs();
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
    cout <<"Thoi gian thuc thi tuan tu Brightness: "<<durationSequence.count()<<"s"<<endl;
    cout <<"Thoi gian thuc thi song song Brightness: "<<durationParallel.count()<<"s"<<endl;
    return result;
}