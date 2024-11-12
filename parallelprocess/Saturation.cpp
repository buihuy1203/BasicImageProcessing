#include <iostream>
#include <opencv2\opencv.hpp>
#include <algorithm>
#include <cmath>
#include <omp.h>
using namespace std;
using namespace cv;
//fasdasdga
template<typename T>
T newclamp(const T& value, const T& low, const T& high) {
    return (value < low) ? low : (value > high) ? high : value;
}

Mat ParallelSaturation(const Mat &input, float set_sar, int process) {
    auto startSequence = chrono::high_resolution_clock::now(); 
    int height = input.rows;
    int width = input.cols;
    Mat result = Mat::zeros(height, width, CV_8UC3);
    int procs_num = process;
    omp_set_num_threads(procs_num);
    auto startParallel = chrono::high_resolution_clock::now(); 
    #pragma omp parallel for collapse(2) shared(input, result, set_sar)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            Vec3b pixel = input.at<Vec3b>(y, x);
            float blue = static_cast<float>(pixel[0]) / 255.0;
            float green = static_cast<float>(pixel[1]) / 255.0;
            float red = static_cast<float>(pixel[2]) / 255.0;

            float max_channel = std::max({blue, green, red});
            float min_channel = std::min({blue, green, red});
            float delta = max_channel - min_channel;
            float light = (max_channel + min_channel) / 2;

            float sar = (delta == 0.0f) ? 0 : delta / (1 - abs(2 * light - 1));
            float hue;
            if (delta == 0) hue = 0;
            else if (max_channel == red) hue = 60 * fmod(((green - blue) / delta), 6);
            else if (max_channel == green) hue = 60 * ((blue - red) / delta + 2);
            else hue = 60 * ((red - green) / delta + 4);
            if (hue < 0) hue += 360;

            float new_sar = std::max(0.0f, std::min(1.0f, set_sar));
            float chroma = (1 - abs(2 * light - 1)) * new_sar;
            float temp = chroma * (1 - abs(fmod((hue / 60.0), 2) - 1));
            float mem = light - chroma / 2;

            float res_red = 0.0f, res_green = 0.0f, res_blue = 0.0f;
            if (hue < 60) { res_red = chroma; res_green = temp; }
            else if (hue < 120) { res_red = temp; res_green = chroma; }
            else if (hue < 180) { res_green = chroma; res_blue = temp; }
            else if (hue < 240) { res_green = temp; res_blue = chroma; }
            else if (hue < 300) { res_red = temp; res_blue = chroma; }
            else { res_red = chroma; res_blue = temp; }

            result.at<Vec3b>(y, x) = Vec3b(
                static_cast<uchar>(newclamp(round((res_blue + mem) * 255), 0.0f, 255.0f)),
                static_cast<uchar>(newclamp(round((res_green + mem) * 255), 0.0f, 255.0f)),
                static_cast<uchar>(newclamp(round((res_red + mem) * 255), 0.0f, 255.0f))
            );
        }
    }
    auto endParralel = chrono::high_resolution_clock::now(); 
    auto endSequence = chrono::high_resolution_clock::now(); 
    chrono::duration<double> durationSequence = endSequence - startSequence;
    chrono::duration<double> durationParallel = endParralel - startParallel;
    cout <<"Thoi gian thuc thi tong chuong trinh Saturation: "<<durationSequence.count()<<"s"<<endl;
    cout <<"Thoi gian thuc thi song song Saturation: "<<durationParallel.count()<<"s"<<endl;
    return result;
}