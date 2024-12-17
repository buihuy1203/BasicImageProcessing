__kernel void sharpen(__global const uchar* inputBuffer, 
                      __global uchar* resultBuffer, 
                      int width, 
                      int height) {
    int j = get_global_id(0);
    int i = get_global_id(1);
    int kernel2D[3][3] = {
            {0, 1, 0},
            {1, -4, 1},
            {0, 1, 0}
        };
    if (i > 0 && i < height - 1 && j > 0 && j < width - 1) {
        int sum = 0;
        for (int k = -1; k <= 1; ++k) {
            for (int l = -1; l <= 1; ++l) {
                if ((i + k) >= 0 && (i + k) < height && (j + l) >= 0 && (j + l) < width) {
                    sum += kernel2D[k + 1][l + 1] * inputBuffer[(i + k) * width + (j + l)];
                }
            }
        }
        resultBuffer[i * width + j] = (uchar)clamp(sum, 0, 255);
    }
}

__kernel void applySharpness(__global const uchar* input, 
                             __global const uchar* result, 
                             __global uchar* sharpResult, 
                             float sharp, 
                             int width, 
                             int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < width && y < height) {
        for (int c = 0; c < 3; c++) {
            int idx = (y * width + x) * 3 + c;
            int grayIdx = y * width + x;
            int newValue = (float)input[idx] - sharp * (float)result[grayIdx];
            sharpResult[idx] = (uchar)clamp(newValue, 0, 255);
        }
    }
}
