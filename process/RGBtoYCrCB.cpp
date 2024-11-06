#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

Mat YCrCBImage(const Mat &input) {
    int height = input.rows;
    int width = input.cols;

    Mat result = Mat::zeros(height, width, CV_8UC3);

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
    return result;  
}