#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

__global__ void process(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, cv::cuda::PtrStep<uchar3> gauss, int rows, int cols)
{
    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    //const int gLoc = threadIdx.x + blockIdx.x * blockDim.x + threadIdx.y * rows + blockIdx.y * blockDim.y * cols;

    uchar3 full = make_uchar3(255, 255, 255);
    float sum = 0;
    float value[] = { 0,0,0 };


    if (dst_x < cols && dst_y < rows)
    {
        float sumBlur = 0;
        uchar3 val = src(dst_y, dst_x);
        for (int i = -2; i <= 2; i++) {
            for (int j = -2; j <= 2; j++) {

                sumBlur += gauss(i, j).x;

                value[0] += gauss(i, j).x * val.x;
                value[1] += gauss(i, j).x * val.y;
                value[2] += gauss(i, j).x * val.z;

            }
        }

        value[0] /= sum;
        value[1] /= sum;
        value[2] /= sum;

        dst(dst_y, dst_x).x = uchar(value[0] );
        dst(dst_y, dst_x).y = uchar(value[1] );
        dst(dst_y, dst_x).z = uchar(value[2] );
    }
    
   

    
}

int divUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void startCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::cuda::GpuMat& gauss)
{
    const dim3 block(32, 8);
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    process <<<grid, block>>> (src, dst, gauss, dst.rows, dst.cols);

}

