void FullTest(){
    BlurMPTime = 0;
    BlurCLTime = 0;
    BrightMPTime = 0;
    BrightCLTime = 0;
    SharpMPTime = 0;
    SharpCLTime = 0;
    YCrCBMPTime = 0;
    YCrCBCLTime = 0;
    SatMPTime = 0;
    SatCLTime = 0;
    BlurSeqTime = 0;
    BrightNessSeqTime = 0;
    SharpSeqTime = 0;
    RGBtoYCrCBSeqTime = 0;
    SatSeqTime = 0;
    ofstream myfile;
    myfile.open ("log.txt", ios_base::app);
    /*auto start1 = chrono::high_resolution_clock::now(); 
    for(int i = 1; i <= 200; i++){
        // Image Path
        string path = "imagetest/meo_xe_tang ";
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

    cout << "Image: "<<i<<" finish"<<endl;
    }
    //End clock sequence
    auto end1 = chrono::high_resolution_clock::now();
    //Check time 
    //imshow("SharpIm", SharpIm);
    chrono::duration<double> duration1 = end1 - start1;
    cout <<"Sequence time: "<<duration1.count()<<"s"<<endl;
    myfile <<"Sequence time total: "<<duration1.count()<<"s"<<endl;
    myfile <<"Sequence time main: "<<(BlurSeqTime + SharpSeqTime + BrightNessSeqTime + SatSeqTime + RGBtoYCrCBSeqTime)<<"s"<<endl;
    //Start new clock
    auto start2 = chrono::high_resolution_clock::now(); 
    //Parallel Processing
    
    for(int i = 1; i <= 200; i++){
        // Image Path
        string path = "imagetest/meo_xe_tang ";
        path += "(";
        path += to_string(i);
        path += ")";
        path += ".jpg";
        Mat image1 = imread(path, IMREAD_COLOR);
        if (image1.empty()) {
            cout << "Can't read image" << endl;
            break;
        }
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

    cout << "Image: "<<i<<"finish"<<endl;
    }
    //End clock
    auto end2 = chrono::high_resolution_clock::now(); 

    //Get OpenMP Processing Time
    chrono::duration<double> duration2 = end2 - start2;
    cout <<"Parallel OpenMP time: "<<duration2.count()<<"s"<<endl;
    myfile <<"Parallel OpenMP time total: "<<duration2.count()<<"s"<<endl;
    myfile <<"Parallel OpenMP time main: "<<(BlurMPTime + SharpMPTime + BrightMPTime + SatMPTime + YCrCBMPTime)<<"s"<<endl;
    *///Start Platform OpenCL
    //Check Device
    
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    for (auto &platform : platforms) {
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        for (auto &device : devices) {
            std::cout << "Platform: " << platform.getInfo<CL_PLATFORM_NAME>() << ", Device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        }
    }
    //Start new clock
    auto start3 = chrono::high_resolution_clock::now(); 
    for(int i = 1; i <= 200; i++){
        // Image Path
        string path = "imagetest/meo_xe_tang ";
        path += "(";
        path += to_string(i);
        path += ")";
        path += ".jpg";
        Mat image1 = imread(path, IMREAD_COLOR);
        if (image1.empty()) {
            cout << "Can't read image" << endl;
            break;
        }

    //Brightness OpenCL
    Mat BrightImCL= ParallelBrightnessOpenCL(image1, -100);

    Mat YCrCBImCL = ParallelYCrCBOpenCL(image1);

    Mat SatImCL = ParallelSaturationOpenCL(image1,1);

    Mat BlurImCL = ParallelBlurOpenCL(image1, 1);

    Mat SharpImCL = ParallelSharpnessOpenCL(image1, 5);

    cout << "Image: "<<i<<"finish"<<endl;
    }
    //End Clock
    auto end3 = chrono::high_resolution_clock::now(); 
    //Get OpenCL Processing Time
    chrono::duration<double> duration3 = end3 - start3;
    cout <<"Parallel OpenCL time: "<<duration3.count()<<"s"<<endl;
    myfile <<"Parallel OpenCL time total: "<<duration3.count()<<"s"<<endl;
    myfile <<"Parallel OpenCL time main: "<<(BlurCLTime + SharpCLTime + BrightCLTime + SatCLTime + YCrCBCLTime)<<"s"<<endl;
    myfile.close();
}