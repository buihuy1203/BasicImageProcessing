__kernel void brightness(__global const uchar *input, __global uchar *output,int rows,int cols, int bright) {
    int y = get_global_id(0);
    int x = get_global_id(1);

    if (y < rows && x < cols) {
        int idx = (y * cols + x) * 3;
        output[idx]= (uchar)clamp(input[idx]+(float)bright,0.0f,255.0f); // Blue
        output[idx + 1] = (uchar)clamp(input[idx+1]+(float)bright,0.0f,255.0f); // Green
        output[idx + 2] = (uchar)clamp(input[idx+2]+(float)bright,0.0f,255.0f); // Red
    }
}