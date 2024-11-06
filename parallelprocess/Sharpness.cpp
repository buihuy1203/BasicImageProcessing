#include <iostream>
#include <opencv2/opencv.hpp>
#include <algorithm> 
#include <cmath>
#include <omp.h>
using namespace std;
using namespace cv;

Mat ParallelSharpness(const Mat &input, double sharp){
    Mat result = Mat::zeros(input.size(), input.type());
    Mat grayImage;
    cvtColor(input, grayImage, COLOR_BGR2GRAY);
    int kernel[3][3] = {
        {0, 1, 0},
        {1, -4, 1},
        {0, 1, 0}
    };
    for (int i = 1; i < grayImage.rows - 1; ++i) {
        for (int j = 1; j < grayImage.cols - 1; ++j) {
            // Tính giá trị mới cho pixel (i, j)
            int sum = 0;
            for (int k = -1; k <= 1; ++k) {
                for (int l = -1; l <= 1; ++l) {
                    sum += kernel[k + 1][l + 1] * grayImage.at<uchar>(i + k, j + l);
                }
            }
            // Giới hạn giá trị để nằm trong khoảng [0, 255]
            result.at<uchar>(i, j) = saturate_cast<uchar>(sum);
        }
    }
    Mat sharpenedImage = Mat::zeros(input.size(), input.type());
    
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            for (int c = 0; c < 3; ++c) { 
                int newValue = input.at<Vec3b>(i, j)[c] - sharp * result.at<uchar>(i, j);
                sharpenedImage.at<Vec3b>(i, j)[c] = saturate_cast<uchar>(newValue);
            }
        }
    }

    return sharpenedImage;
}
