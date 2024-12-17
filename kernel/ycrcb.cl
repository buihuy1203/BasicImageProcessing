__kernel void ycrcb(__global const uchar *input,
                         __global uchar *output,
                         int cols,
                         int rows) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if ( x < cols && y < rows ) {
        int idx = (y * cols + x) * 3;
        // Input Value
        uchar blue = input[idx];
        uchar green = input[idx + 1];
        uchar red = input[idx + 2];

        // RGB to YCrCB
        uchar Y = (uchar)(0.299f * red + 0.587f * green + 0.114f * blue);
        uchar Cb = (uchar)(128.0f + (blue - Y) * 0.564f);
        uchar Cr = (uchar)(128.0f + (red - Y) * 0.713f);

        // Output Value
        output[idx]     = Y;
        output[idx + 1] = Cb;
        output[idx + 2] = Cr;
    }
}