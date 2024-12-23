#include <iostream>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <chrono>
#include <CL/cl.hpp>
#include <fstream>
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
#include "testscene/fulltest.cpp"
#include "testscene/checktest.cpp"
#include "testscene/invidualtest.cpp"
using namespace std;
using namespace cv;

int main() {
    cout << "Start Full Test" << endl;
    FullTest();
    //cout << "Start Check Test" << endl;
    //checktest();
    cout << "Start Bright Test" << endl;
    BrightTest();
    cout << "Start Saturation Test" << endl;
    SatTest();
    cout << "Start Sharpness Test" << endl;
    SharpTest();
    cout << "Start Blur Test" << endl;
    BlurTest();
    cout << "Start YCrCB Test" << endl;
    YCrCbTest();
    return 0;
}