#include <iostream>
#include <opencv2/opencv.hpp>
#include <algorithm> 
#include <cmath>
#include <omp.h>
using namespace std;
using namespace cv;
template<typename T>
T newclamp(const T& value, const T& low, const T& high) {
    return (value < low) ? low : (value > high) ? high : value;
}
Mat ParallelSaturation(const Mat &input, float set_sar){
    int height = input.rows;
    int width = input.cols;

    Mat result = Mat::zeros(height, width, CV_8UC3);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            Vec3b pixel = input.at<Vec3b>(y, x);
            float blue = static_cast<float>(pixel[0])/255.0;
            float green = static_cast<float>(pixel[1])/255.0;
            float red = static_cast<float>(pixel[2])/255.0;

            float max_channel = std::max({blue, green, red});
            float min_channel = std::min({blue, green, red});
            float delta = max_channel-min_channel;
            
            float light = (max_channel+min_channel)/2;
            float sar;
            if(delta == 0.0f)
                sar = 0;
            else
                sar = delta/(1-abs(2*light-1));
            float hue;
            if (delta == 0)
                hue = 0; // Undefined hue khi delta = 0
            else if(max_channel == red)
                hue = 60*static_cast<float>(fmod(((green-blue)/delta),6));
            else if(max_channel == green)
                hue = 60*((blue-green)/delta + 2);
            else if(max_channel == blue)
                hue = 60*((red-green)/delta + 4);
            if (hue < 0) hue += 360;
            float new_sar = std::max(0.0f, std::min(1.0f, set_sar));

            float chroma = (1-abs(2*light-1))*new_sar;
            float temp = chroma*(1-abs(static_cast<float>(fmod((hue/60.0),2))-1));
            float mem = light-chroma/2;

            float res_red =0.0f, res_green = 0.0f, res_blue = 0.0f;
            if(hue < 60&&hue>=0){
                res_red=chroma;
                res_green=temp;
                res_blue = 0;
            }else if(hue <120 && hue >=60){
                res_red=temp;
                res_green=chroma;
                res_blue = 0;
            }else if(hue >=120 && hue <180){
                res_red=0;
                res_green=chroma;
                res_blue = temp;
            }else if(hue >= 180 && hue <240){
                res_red=0;
                res_green=temp;
                res_blue = chroma;
            }else if(hue >= 240 && hue <300){
                res_red=temp;
                res_green=0;
                res_blue = chroma;
            }else if(hue>=300 && hue <360){
                res_red=chroma;
                res_green=0;
                res_blue = temp;
            }
            result.at<Vec3b>(y, x) = Vec3b(\
                static_cast<uchar>(newclamp(round((res_blue + mem) * 255), 0.0f, 255.0f)),\
                static_cast<uchar>(newclamp(round((res_green + mem) * 255), 0.0f, 255.0f)),\
                static_cast<uchar>(newclamp(round((res_red + mem) * 255), 0.0f, 255.0f))\
            );
        }
    }
    return result;  
}