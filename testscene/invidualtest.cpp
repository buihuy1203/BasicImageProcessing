void YCrCbTest(){
    ofstream myfile;
    myfile.open ("log.txt");
    auto start1 = chrono::high_resolution_clock::now(); 
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

    cout << "Image: "<<i<<" finish"<<endl;
    }
    //End clock sequence
    auto end1 = chrono::high_resolution_clock::now();
    //Check time 
    chrono::duration<double> duration1 = end1 - start1;
    cout <<"Sequence time: "<<duration1.count()<<"s"<<endl;
    myfile <<"Sequence time YCrCB: "<<duration1.count()<<"s"<<endl;
    //Start new clock
    auto start2 = chrono::high_resolution_clock::now(); 
    //Parallel Processing
    
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
        int processor = 4;
    //YCrCB OpenMP
    Mat YCrCBImP= ParallelYCrCBImage(image1,processor);

    cout << "Image: "<<i<<"finish"<<endl;
    }
    //End clock
    auto end2 = chrono::high_resolution_clock::now(); 

    //Get OpenMP Processing Time
    chrono::duration<double> duration2 = end2 - start2;
    cout <<"Parallel OpenMP time: "<<duration2.count()<<"s"<<endl;
    myfile <<"Parallel OpenMP time YCrCB: "<<duration2.count()<<"s"<<endl;
    
    //Start Platform OpenCL
    //Check Device
    
    //listPlatformsAndDevices();

    //Start new clock
    auto start3 = chrono::high_resolution_clock::now(); 
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

    Mat YCrCBImCL = ParallelYCrCBOpenCL(image1);

    cout << "Image: "<<i<<"finish"<<endl;
    }
    //End Clock
    auto end3 = chrono::high_resolution_clock::now(); 
    //Get OpenCL Processing Time
    chrono::duration<double> duration3 = end3 - start3;
    cout <<"Parallel OpenCL time YCrCb: "<<duration3.count()<<"s"<<endl;
    myfile <<"Parallel OpenCL time YCrCb: "<<duration3.count()<<"s"<<endl;
    myfile.close();
    // Read Image
}

void BrightTest(){
    ofstream myfile;
    myfile.open ("log.txt");
    auto start1 = chrono::high_resolution_clock::now(); 
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

    // Brightness
    Mat BrightIm= BrightNess(image1, -100);

    cout << "Image: "<<i<<" finish"<<endl;
    }
    //End clock sequence
    auto end1 = chrono::high_resolution_clock::now();
    //Check time 
    chrono::duration<double> duration1 = end1 - start1;
    cout <<"Sequence time Bright: "<<duration1.count()<<"s"<<endl;
    myfile <<"Sequence time Bright: "<<duration1.count()<<"s"<<endl;
    //Start new clock
    auto start2 = chrono::high_resolution_clock::now(); 
    //Parallel Processing
    
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
        int processor = 4;

    //Brightness OpenMP
    Mat BrightImP= ParallelBrightNess(image1, -100,processor);

    cout << "Image: "<<i<<"finish"<<endl;
    }
    //End clock
    auto end2 = chrono::high_resolution_clock::now(); 

    //Get OpenMP Processing Time
    chrono::duration<double> duration2 = end2 - start2;
    cout <<"Parallel OpenMP time Bright: "<<duration2.count()<<"s"<<endl;
    myfile <<"Parallel OpenMP time Bright: "<<duration2.count()<<"s"<<endl;
    //Start Platform OpenCL
    //Check Device
    
    //listPlatformsAndDevices();

    //Start new clock
    auto start3 = chrono::high_resolution_clock::now(); 
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

    //Brightness OpenCL
    Mat BrightImCL= ParallelBrightnessOpenCL(image1, -100);

    cout << "Image: "<<i<<"finish"<<endl;
    }
    //End Clock
    auto end3 = chrono::high_resolution_clock::now(); 
    //Get OpenCL Processing Time
    chrono::duration<double> duration3 = end3 - start3;
    cout <<"Parallel OpenCL time Bright: "<<duration3.count()<<"s"<<endl;
    myfile <<"Parallel OpenCL time Bright: "<<duration3.count()<<"s"<<endl;
    myfile.close();
}

void SharpTest(){
    ofstream myfile;
    myfile.open ("log.txt");
    auto start1 = chrono::high_resolution_clock::now(); 
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

    // Sharpness
    Mat SharpIm= Sharpness(image1, 5);

    cout << "Image: "<<i<<" finish"<<endl;
    }
    //End clock sequence
    auto end1 = chrono::high_resolution_clock::now();
    //Check time 
    //imshow("SharpIm", SharpIm);
    chrono::duration<double> duration1 = end1 - start1;
    cout <<"Sequence time Sharp: "<<duration1.count()<<"s"<<endl;
    myfile <<"Sequence time Sharp: "<<duration1.count()<<"s"<<endl;

    //Start new clock
    auto start2 = chrono::high_resolution_clock::now(); 
    //Parallel Processing
    
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
        int processor = 4;

    //Sharpness OpenMP
    Mat SharpImP= ParallelSharpness(image1, 5,processor);

    cout << "Image: "<<i<<"finish"<<endl;
    }
    //End clock
    auto end2 = chrono::high_resolution_clock::now(); 

    //Get OpenMP Processing Time
    chrono::duration<double> duration2 = end2 - start2;
    cout <<"Parallel OpenMP time Sharp: "<<duration2.count()<<"s"<<endl;
    myfile <<"Parallel OpenMP time Sharp: "<<duration2.count()<<"s"<<endl;
    //Start Platform OpenCL
    //Check Device
    
    //listPlatformsAndDevices();

    //Start new clock
    auto start3 = chrono::high_resolution_clock::now(); 
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
    //Sharpness OpenCL
    Mat SharpImCL = ParallelSharpnessOpenCL(image1, 5);

    cout << "Image: "<<i<<"finish"<<endl;
    }
    //End Clock
    auto end3 = chrono::high_resolution_clock::now(); 
    //Get OpenCL Processing Time
    chrono::duration<double> duration3 = end3 - start3;
    cout <<"Parallel OpenCL time Sharp: "<<duration3.count()<<"s"<<endl;
    myfile <<"Parallel OpenCL time Sharp: "<<duration3.count()<<"s"<<endl;
    myfile.close();
}

void SatTest(){
    ofstream myfile;
    myfile.open ("log.txt");
    auto start1 = chrono::high_resolution_clock::now(); 
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

    // Saturation
    Mat SatIm= Saturation(image1, 1);

    cout << "Image: "<<i<<" finish"<<endl;
    }
    //End clock sequence
    auto end1 = chrono::high_resolution_clock::now();
    //Check time 
    //imshow("SharpIm", SharpIm);
    chrono::duration<double> duration1 = end1 - start1;
    cout <<"Sequence time Sat: "<<duration1.count()<<"s"<<endl;
    myfile <<"Sequence time Sat: "<<duration1.count()<<"s"<<endl;

    //Start new clock
    auto start2 = chrono::high_resolution_clock::now(); 
    //Parallel Processing
    
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
        int processor = 4;
    
    //Saturation OpenMP
    Mat SatImP= ParallelSaturation(image1, 1, processor);

    cout << "Image: "<<i<<"finish"<<endl;
    }
    //End clock
    auto end2 = chrono::high_resolution_clock::now(); 

    //Get OpenMP Processing Time
    chrono::duration<double> duration2 = end2 - start2;
    cout <<"Parallel OpenMP time: "<<duration2.count()<<"s"<<endl;
    myfile <<"Parallel OpenMP time Sat: "<<duration2.count()<<"s"<<endl;
    //Start Platform OpenCL
    //Check Device
    
    //listPlatformsAndDevices();

    //Start new clock
    auto start3 = chrono::high_resolution_clock::now(); 
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

    //Saturation OpenCL

    Mat SatImCL = ParallelSaturationOpenCL(image1,1);

    cout << "Image: "<<i<<"finish"<<endl;
    }
    //End Clock
    auto end3 = chrono::high_resolution_clock::now(); 
    //Get OpenCL Processing Time
    chrono::duration<double> duration3 = end3 - start3;
    cout <<"Parallel OpenCL time Sat:"<<duration3.count()<<"s"<<endl;
    myfile <<"Parallel OpenCL time Sat: "<<duration3.count()<<"s"<<endl;
    myfile.close();
}

void BlurTest(){
    ofstream myfile;
    myfile.open ("log.txt");
    auto start1 = chrono::high_resolution_clock::now(); 
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

    // Bluring
    Mat BlurIm= BlurImage(image1, 1);

    cout << "Image: "<<i<<" finish"<<endl;
    }
    //End clock sequence
    auto end1 = chrono::high_resolution_clock::now();
    //Check time 
    //imshow("SharpIm", SharpIm);
    chrono::duration<double> duration1 = end1 - start1;
    cout <<"Sequence time Blur: "<<duration1.count()<<"s"<<endl;
    myfile <<"Sequence time Blur: "<<duration1.count()<<"s"<<endl;
    //Start new clock
    auto start2 = chrono::high_resolution_clock::now(); 
    //Parallel Processing
    
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
        int processor = 4;

    //Bluring OpenMP
    Mat BlurImP= ParallelBlurImage(image1, 1,processor);

    cout << "Image: "<<i<<"finish"<<endl;
    }
    //End clock
    auto end2 = chrono::high_resolution_clock::now(); 

    //Get OpenMP Processing Time
    chrono::duration<double> duration2 = end2 - start2;
    cout <<"Parallel OpenMP time Blur: "<<duration2.count()<<"s"<<endl;
    myfile <<"Parallel OpenMP time Blur: "<<duration2.count()<<"s"<<endl;
    //Start Platform OpenCL
    //Check Device
    
    //listPlatformsAndDevices();

    //Start new clock
    auto start3 = chrono::high_resolution_clock::now(); 
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
    //Blur OpenCL

    Mat BlurImCL = ParallelBlurOpenCL(image1, 1);

    cout << "Image: "<<i<<"finish"<<endl;
    }
    //End Clock
    auto end3 = chrono::high_resolution_clock::now(); 
    //Get OpenCL Processing Time
    chrono::duration<double> duration3 = end3 - start3;
    cout <<"Parallel OpenCL time Blur: "<<duration3.count()<<"s"<<endl;
    myfile <<"Parallel OpenCL time Blur: "<<duration3.count()<<"s"<<endl;
    myfile.close();
}