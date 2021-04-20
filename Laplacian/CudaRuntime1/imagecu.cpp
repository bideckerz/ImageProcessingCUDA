#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <chrono>  // for high_resolution_clock
# define M_PI           3.14159265358979323846
using namespace std;

void startCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst);


namespace psrs
{
    void AddFrame(const cv::Mat_<cv::Vec3b>& src, cv::Mat_<cv::Vec3b>& dst, int fSize)
    {
        dst.create(src.rows + 2 * fSize, src.cols + 2 * fSize);
        dst = cv::Vec3b(0, 0, 0);
        for (int i = 0; i < src.rows; i++)
            for (int j = 0; j < src.cols; j++)
                for (int c = 0; c < 3; c++)
                    dst(i + fSize, j + fSize)[c] = src(i, j)[c];
    }
  
}

int main(int argc, char** argv)
{
    cv::namedWindow("Original Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Processed Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);

    cv::Mat h_img = cv::imread(argv[1]);
    cv::cuda::GpuMat d_img, d_result;
    cv::Mat_<cv::Vec3b> h_result;



    d_img.upload(h_img);
    d_result.upload(h_img);

    cv::imshow("Original Image", d_img);
    int frameSize = atoi(argv[2]);
    psrs::AddFrame(h_img, h_result, frameSize);


    auto begin = chrono::high_resolution_clock::now();
    const int iter = 10000;

        
    for (int i = 0; i < iter; i++)
    {
        startCUDA(d_img, d_result);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - begin;

    cv::imshow("Processed Image", h_result);

    cout << diff.count() << endl;
    cout << diff.count() / iter << endl;
    cout << iter / diff.count() << endl;

    cv::waitKey();
    return 0;

    return 0;
}
