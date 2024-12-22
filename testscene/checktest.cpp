bool compare(const Mat &input, const Mat &result){
    for(int x = 0; x < input.rows; x++){
        for(int y = 0; y < input.cols; y++){
            Vec3b pixelin = input.at<Vec3b>(x, y);
            Vec3b pixelres = result.at<Vec3b>(x, y); 
            if (pixelin != pixelres) {
                return false;
            }
        }
    }
    return true;
}

void checktest(){
    for(int i = 1; i <= 200; i++){
        // Image Path
        string path = "E:/Bai Tap/Lap trinh song song/ImageProcessing/imagetest/meo_xe_tang ";
        path += "(";
        path += to_string(i);
        path += ")";
        path += ".jpg";
        Mat image1 = imread(path, IMREAD_COLOR);
        if (image1.empty()) {
            cout << "Can't read image" << endl;
            break;
        }
    //imshow("Original", image1);
    
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
    
    int processor = 4;
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

    //Brightness OpenCL
    Mat BrightImCL= ParallelBrightnessOpenCL(image1, -100);

    Mat YCrCBImCL = ParallelYCrCBOpenCL(image1);

    Mat SatImCL = ParallelSaturationOpenCL(image1,1);

    Mat BlurImCL = ParallelBlurOpenCL(image1, 1);

    Mat SharpImCL = ParallelSharpnessOpenCL(image1, 5);
    //Check Image
    //imshow("YCrCBImCL", YCrCBImCL);
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
    cout << "Image: "<<i<<" finish"<<endl;
    }
}