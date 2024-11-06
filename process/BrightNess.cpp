#include <iostream>
#include <opencv2/opencv.hpp>
#include <algorithm> 
#include <cmath>

using namespace std;
using namespace cv;

Mat BrightNess(Mat &input, int bright){
    Mat result =  Mat::zeros(input.rows, input.cols, CV_8UC3);
    for(int y=0; y < input.rows; y++){
        for(int x = 0; x < input.cols; x++){
            Vec3b pixel = input.at<Vec3b>(y,x);
            uchar blue = pixel[0];
            uchar green = pixel[1];
            uchar red = pixel[2]; 
            result.at<Vec3b>(y, x) = Vec3b(saturate_cast<uchar>(blue+bright), saturate_cast<uchar>(green+bright), saturate_cast<uchar>(red+bright));
        }
    }
    return result;
}