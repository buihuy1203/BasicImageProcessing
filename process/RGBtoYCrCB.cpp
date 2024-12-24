#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
using namespace std;
using namespace cv;

static double RGBtoYCrCBSeqTime = 0;

Mat YCrCBImage(const Mat &input) {
    int height = input.rows;
    int width = input.cols;

    Mat result = Mat::zeros(height, width, CV_8UC3);
    auto start = chrono::high_resolution_clock::now();
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
    chrono::duration<double> duration = end - start;
    RGBtoYCrCBSeqTime += duration.count();
    return result;  
}