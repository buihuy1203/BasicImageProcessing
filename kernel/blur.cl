__kernel void gaussblur(__global const uchar* input, 
                   __global uchar* output, 
                   __global const float* kernel1D,
                   int rows, 
                   int cols,
                   int channels,
                   int kernelSize) {
    int x = get_global_id(0); // pixel row
    int y = get_global_id(1); // pixel col
    int halfSize = kernelSize/2;

    if (x < rows && y < cols) {
        float sum[3] = {0.0f, 0.0f, 0.0f};

        for (int kx = -halfSize; kx <= halfSize; ++kx) {
            for (int ky = -halfSize; ky <= halfSize; ++ky) {
                int px = clamp(x + kx, 0, rows - 1);
                int py = clamp(y + ky, 0, cols - 1);

                int pixelIndex = (px * cols + py) * channels;
                float weight = kernel1D[(kx + halfSize) * kernelSize + (ky + halfSize)];

                sum[0] += (float)input[pixelIndex] * weight;       // Blue channel
                sum[1] += (float)input[pixelIndex + 1] * weight;   // Green channel
                sum[2] += (float)input[pixelIndex + 2] * weight;   // Red channel
            }
        }

        int outputIndex = (x * cols + y) * channels;
        output[outputIndex] = (uchar)clamp(sum[0], 0.0f, 255.0f);
        output[outputIndex + 1] = (uchar)clamp(sum[1], 0.0f, 255.0f);
        output[outputIndex + 2] = (uchar)clamp(sum[2], 0.0f, 255.0f);
    }
}