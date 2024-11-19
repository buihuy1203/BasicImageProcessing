__kernel void brightness(__global const uchar *input,
                         __global uchar *output,
                         const int rows,
                         const int cols,
                         const int bright) {
    int y = get_global_id(0);
    int x = get_global_id(1);

    if (y < rows && x < cols) {
        int idx = (y * cols + x) * 3;
        output[idx]     = clamp(input[idx]     + bright, 0, 255);
        output[idx + 1] = clamp(input[idx + 1] + bright, 0, 255);
        output[idx + 2] = clamp(input[idx + 2] + bright, 0, 255);
    }
}