#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

__global__ void process(const cv::cuda::PtrStep<uchar3> src1, const cv::cuda::PtrStep<uchar3> src2, cv::cuda::PtrStep<uchar3> dsta, cv::cuda::PtrStep<uchar3> dsts, cv::cuda::PtrStep<uchar3> dstd, cv::cuda::PtrStep<uchar3> dstm, int rows, int cols )
{
 
  const int dsta_x = blockDim.x * blockIdx.x + threadIdx.x;
  const int dsta_y = blockDim.y * blockIdx.y + threadIdx.y;

  const int dsts_x = blockDim.x * blockIdx.x + threadIdx.x;
  const int dsts_y = blockDim.y * blockIdx.y + threadIdx.y;

  const int dstd_x = blockDim.x * blockIdx.x + threadIdx.x;
  const int dstd_y = blockDim.y * blockIdx.y + threadIdx.y;

  const int dstm_x = blockDim.x * blockIdx.x + threadIdx.x;
  const int dstm_y = blockDim.y * blockIdx.y + threadIdx.y;
  
  if (dsta_x < cols && dsta_y < rows)
    {
      uchar3 val = src1(dsta_y, dsta_x);
      uchar3 val2 = src2(dsta_y, dsta_x);

      dsta(dsta_y, dsta_x).x = (val.x + val2.x)/2;
      dsta(dsta_y, dsta_x).y = (val.y + val2.y) / 2;
      dsta(dsta_y, dsta_x).z = (val.z + val2.z) / 2;
          
    }

  if (dsts_x < cols && dsts_y < rows) {
      uchar3 val = src1(dsts_y, dsts_x);
      uchar3 val2 = src2(dsts_y, dsts_x);

      dsts(dsts_y, dsts_x).x = (val.x - val2.x);
      dsts(dsts_y, dsts_x).y = (val.y - val2.y);
      dsts(dsts_y, dsts_x).z = (val.z - val2.z);

  }
  if (dstd_x < cols && dstd_y < rows) {
      uchar3 val = src1(dstd_y, dstd_x);
      uchar3 val2 = src2(dstd_y, dstd_x);

      dstd(dstd_y, dstd_x).x = ((val2.x / val.x) * 50) + 20;
      dstd(dstd_y, dstd_x).y = ((val2.y / val.y) * 50) + 20;
      dstd(dstd_y, dstd_x).z = ((val2.z / val.z) * 50) + 20;

  }
  if (dstm_x < cols && dstm_y < rows) {
      uchar3 val = src1(dstm_y, dstm_x);
      uchar3 val2 = src2(dstm_y, dstm_x);

      dstm(dstm_y, dstm_x).x = (val.x * val2.x) / 255;
      dstm(dstm_y, dstm_x).y = (val.y * val2.y) / 255;
      dstm(dstm_y, dstm_x).z = (val.z * val2.z) / 255;

  }
}

int divUp(int a, int b)
{
  return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void addCUDA ( cv::cuda::GpuMat& src1, cv::cuda::GpuMat& src2,  cv::cuda::GpuMat& dsta, cv::cuda::GpuMat& dsts, cv::cuda::GpuMat& dstd, cv::cuda::GpuMat& dstm)
{
  const dim3 block(32, 8);
  const dim3 grid(divUp(dsta.cols, block.x), divUp(dsta.rows, block.y));

  process<<<grid, block>>>(src1, src2, dsta,dsts, dstd, dstm, dsta.rows, dsta.cols);

}

