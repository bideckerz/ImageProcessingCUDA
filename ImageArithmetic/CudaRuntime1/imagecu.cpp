#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <chrono>  // for high_resolution_clock

using namespace std;

void addCUDA ( cv::cuda::GpuMat& src1, cv::cuda::GpuMat& src2, cv::cuda::GpuMat& dsta, cv::cuda::GpuMat& dsts, cv::cuda::GpuMat& dstd, cv::cuda::GpuMat& dstm);

int main( int argc, char** argv )
{
  cv::namedWindow("Original Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);
  cv::namedWindow("Another Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);
  cv::namedWindow("Added Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);
  cv::namedWindow("Substracted Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);
  cv::namedWindow("Divided Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);
  cv::namedWindow("Multiplied Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);


  cv::Mat h_img = cv::imread(argv[1]);
  cv::Mat h_img1 = cv::imread(argv[2]);
  cv::cuda::GpuMat d_img, d_img1, d_resulta, d_results, d_resultd, d_resultm;
  cv::Mat h_result;


  d_img.upload(h_img);
  d_img1.upload(h_img1);
  d_resulta.upload(h_img);
  d_results.upload(h_img);
  d_resultd.upload(h_img);
  d_resultm.upload(h_img);


  cv::imshow("Original Image", d_img);
  cv::imshow("Another Image", d_img1);

  
  auto begin = chrono::high_resolution_clock::now();
  const int iter = 10000;

  
  for (int i=0;i<iter;i++)
    {
      addCUDA ( d_img,d_img1,d_resulta, d_results, d_resultd, d_resultm);

    }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end-begin;

  cv::imshow("Added Image", d_resulta);
  cv::imshow("Substracted Image", d_results);
  cv::imshow("Divided Image", d_resultd);
  cv::imshow("Multiplied Image", d_resultm);


  cout << diff.count() << endl;
  cout << diff.count()/iter << endl;
  cout << iter/diff.count() << endl;
  
  cv::waitKey();
  
  return 0;
}
