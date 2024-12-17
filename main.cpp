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

bool compare(const Mat &input, const Mat &result){
    for(int x = 0; x < input.rows; x++){
        for(int y = 0; y < input.cols; y++){
            Vec3b pixelin = input.at<Vec3b>(x, y);
            Vec3b pixelres = result.at<Vec3b>(x, y); 
            if (pixelin != pixelres) {
                cout << "Error at x: " << x << " y: " << y << endl;
                return false;
            }
        }
    }
    return true;
}

int main() {
    // Image Path
    string path = "E:/Bai Tap/Lap trinh song song/ImageProcessing/meo_xe_tang.jpg";

    // Read Image
    Mat image1 = imread(path, IMREAD_COLOR);
    // Check if image is okay
    if (image1.empty()) {
        cout << "Can't read image" << endl;
        return -1;
    }
    //Set Processor
    int processor = 4;
    imshow("Original", image1);
    //Start Clock On Sequence
    auto start1 = chrono::high_resolution_clock::now(); 
    // YCrCb
    Mat YCrCBIm = YCrCBImage(image1);

    // Saturation
    Mat SatIm= Saturation(image1, 1);

    // Sharpness
    Mat SharpIm= Sharpness(image1, 5);

    // Bluring
    Mat BlurIm= BlurImage(image1, 1);

    // Brightness
    Mat BrightIm= BrightNess(image1, -100);

    //End clock sequence
    auto end1 = chrono::high_resolution_clock::now();
    //Check time 
    //imshow("SharpIm", SharpIm);
    chrono::duration<double> duration1 = end1 - start1;
    cout <<"Sequence time: "<<duration1.count()<<"s"<<endl;

    //Start new clock
    auto start2 = chrono::high_resolution_clock::now(); 
    //Parallel Processing

    //YCrCB OpenMP
    Mat YCrCBImP= ParallelYCrCBImage(image1,processor);
    
    //Saturation OpenMP
    Mat SatImP= ParallelSaturation(image1, 1, processor);

    //Sharpness OpenMP
    Mat SharpImP= ParallelSharpness(image1, 5,processor);

    //Bluring OpenMP
    Mat BlurImP= ParallelBlurImage(image1, 1,processor);

    //Brightness OpenMP
    Mat BrightImP= ParallelBrightNess(image1, -100,processor);

    //End clock
    auto end2 = chrono::high_resolution_clock::now(); 

    //Get OpenMP Processing Time
    chrono::duration<double> duration2 = end2 - start2;
    cout <<"Parallel OpenMP time: "<<duration2.count()<<"s"<<endl;
    //Compare Result
    if(compare(YCrCBIm,YCrCBImP)){
        cout <<"YCrCB OpenMP true"<<endl;
    }else
    {
        cout <<"YCrCB OpenMP false"<<endl;
    }
    if(compare(SatIm,SatImP)){
        cout <<"Saturation OpenMP true"<<endl;
    }else
    {
        cout <<"Saturation OpenMP false"<<endl;
    }
    if(compare(BlurIm,BlurImP)){
        cout <<"Bluring OpenMP true"<<endl;
    }else
    {
        cout <<"Bluring OpenMP false"<<endl;
    }
    if(compare(BrightIm,BrightImP)){
        cout <<"Brightness OpenMP true"<<endl;
    }else
    {
        cout <<"Brightness OpenMP false"<<endl;
    }
    if(compare(SharpIm,SharpImP)){
        cout <<"Sharpness OpenMP true"<<endl;
    }else
    {
        cout <<"Sharpness OpenMP false"<<endl;
    }
    //Start Platform OpenCL
    //Check Device
    
    //listPlatformsAndDevices();

    //Start new clock
    auto start3 = chrono::high_resolution_clock::now(); 

    //Brightness OpenCL
    Mat BrightImCL= ParallelBrightnessOpenCL(image1, -100);

    Mat YCrCBImCL = ParallelYCrCBOpenCL(image1);

    Mat SatImCL = ParallelSaturationOpenCL(image1,1);

    Mat BlurImCL = ParallelBlurOpenCL(image1, 1);

    Mat SharpImCL = ParallelSharpnessOpenCL(image1, 5);
    //End Clock
    auto end3 = chrono::high_resolution_clock::now(); 
    //Check Image
    imshow("YCrCBImCL", YCrCBImCL);
    //imshow("BrightImCL", BrightImCL);
    //imshow("SatImCL", SatImCL);
    //imshow("BlurImCL", BlurImCL);
    //imshow("SharpImCL", SharpImCL);
    if(compare(YCrCBIm,YCrCBImCL)){
        cout <<"YCrCB OpenCL true"<<endl;
    }else
    {
        cout <<"YCrCB OpenCL false"<<endl;
    }
    if(compare(BrightIm,BrightImCL)){
        cout <<"Brightness OpenCL true"<<endl;
    }else
    {
        cout <<"Brightness OpenCL false"<<endl;
    }
    if(compare(SatIm,SatImCL)){
        cout <<"Saturation OpenCL true"<<endl;
    }else
    {
        cout <<"Saturation OpenCL false"<<endl;
    }
    if(compare(BlurIm,BlurImCL)){
        cout <<"Blur OpenCL true"<<endl;
    }else
    {
        cout <<"Blur OpenCL false"<<endl;
    }
    if(compare(SharpIm,SharpImCL)){
        cout <<"Sharp OpenCL true"<<endl;
    }else
    {
        cout <<"Sharp OpenCL false"<<endl;
    }

    //Get OpenCL Processing Time
    chrono::duration<double> duration3 = end3 - start3;
    cout <<"Thoi gian thuc thi song song openCL: "<<duration3.count()<<"s"<<endl;
    waitKey(0);
    return 0;
}