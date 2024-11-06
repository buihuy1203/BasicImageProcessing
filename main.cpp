#include <iostream>
#include <opencv2/opencv.hpp>
#include "process/RGBtoYCrCB.cpp"
#include "process/Blur.cpp"
#include "process/BrightNess.cpp"
#include "process/Saturation.cpp"
#include "process/Sharpness.cpp"
using namespace std;
using namespace cv;
int main() {
    // Đường dẫn đến ảnh
    string path = "E:/Bai Tap/Lap trinh song song/ImageProcessing/meo_xe_tang.jpg";

    // Đọc ảnh
    Mat image1 = imread(path, IMREAD_COLOR);
    // Kiểm tra nếu ảnh được đọc thành công
    if (image1.empty()) {
        cout << "Khong the doc anh." << endl;
        return -1;
    }

    // Hiển thị ảnh
    imshow("Original", image1);
    imshow("Saturation image", Saturation(image1, 1));
    imshow("Sharpness image", Sharpness(image1, 10));
    imshow("Blur image", BlurImage(image1, 1));
    imshow("Brightness image", BrightNess(image1, -100));
    waitKey(0);

    return 0;
}