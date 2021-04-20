#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

__global__ void process(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<float3> dst, int rows, int cols)
{
    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    //const int gLoc = threadIdx.x + blockIdx.x * blockDim.x + threadIdx.y * rows + blockIdx.y * blockDim.y * cols;

    uchar3 full = make_uchar3(255, 255, 255);
    float sum = 0;
    
     

    if (dst_x < cols && dst_y < rows)
    {
        float temp[] = { 0,0,0 };
        uchar3 val = src(dst_y, dst_x);
        for (int i = -1; i = 1; i++){
            for (int j = -1; j = 1; j++) {
                temp[0] = val.x * -1;
                temp[1] = val.y * -1;
                temp[2] = val.z * -1;    
            }
        }
        dst(dst_y, dst_x).x = uchar(temp[0] * (val.x*9));
        dst(dst_y, dst_x).y = uchar(temp[1] * (val.y*9));
        dst(dst_y, dst_x).z = uchar(temp[2] * (val.z*9));
    }
    
}

int divUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void startCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst)
{
    const dim3 block(32, 8);
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    process <<<grid, block>>> (src, dst, dst.rows, dst.cols);

}

