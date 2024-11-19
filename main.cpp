#include <iostream>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <chrono>
#include <CL/cl.hpp>
#include "process/RGBtoYCrCB.cpp"
#include "process/Blur.cpp"
#include "process/BrightNess.cpp"
#include "process/Saturation.cpp"
#include "process/Sharpness.cpp"
#include "parallelprocess/RGBtoYCrCB.cpp"
#include "parallelprocess/Blur.cpp"
#include "parallelprocess/BrightNess.cpp"
#include "parallelprocess/Saturation.cpp"
#include "parallelprocess/Sharpness.cpp"
using namespace std;
using namespace cv;

int main() {
    // Đường dẫn đến ảnh
    string path = "E:/Bai Tap/Lap trinh song song/ImageProcessing/meo_xe_tang_co_lon.jpg";

    // Đọc ảnh
    Mat image1 = imread(path, IMREAD_COLOR);
    // Kiểm tra nếu ảnh được đọc thành công
    if (image1.empty()) {
        cout << "Khong the doc anh." << endl;
        return -1;
    }
    imshow("Original", image1);
    
    auto start1 = chrono::high_resolution_clock::now(); 
    // Hiển thị ảnh
    imshow("RGBtoYCrCb image", YCrCBImage(image1));
    imshow("Saturation image", Saturation(image1, 1));
    imshow("Sharpness image", Sharpness(image1, 1));
    imshow("Blur image", BlurImage(image1, 1));
    imshow("Brightness image", BrightNess(image1, -100));
    auto end1 = chrono::high_resolution_clock::now(); 
    chrono::duration<double> duration1 = end1 - start1;
    cout <<"Thoi gian thuc thi tuan tu: "<<duration1.count()<<"s"<<endl;
    int processor = 4;
    auto start2 = chrono::high_resolution_clock::now(); 
    // Hiển thị ảnh
    imshow("RGBtoYCrCb image parallel", ParallelYCrCBImage(image1,processor));
    imshow("Saturation image parallel", ParallelSaturation(image1, 1, processor));
    imshow("Sharpness image parallel", ParallelSharpness(image1, 1,processor));
    imshow("Blur image parallel", ParallelBlurImage(image1, 1,processor));
    imshow("Brightness image parallel", ParallelBrightNess(image1, -100,processor));
    auto end2 = chrono::high_resolution_clock::now(); 
    chrono::duration<double> duration2 = end2 - start2;
    cout <<"Thoi gian thuc thi song song: "<<duration2.count()<<"s"<<endl;
    waitKey(0);
    /*imshow("Brightness image parallel", ParallelBrightNessOpenCL(image1, -100));*/

    return 0;
}