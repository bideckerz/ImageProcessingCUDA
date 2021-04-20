#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <chrono>  // for high_resolution_clock
# define M_PI           3.14159265358979323846
using namespace std;

void startCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::cuda::GpuMat& gauss);


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

    void FilterCreation(double GKernel[][5],int kernelsize)
{
    // intialising standard deviation to 1.0
    double sigma = 5.0;
    double r, s = 2.0 * sigma * sigma;
  
    // sum is for normalization
    double sum = 0.0;
  
    // generating 5x5 kernel
    for (int x = -kernelsize/2; x <= kernelsize/2; x++) {
        for (int y = -kernelsize / 2; y <= kernelsize / 2; y++) {
            r = sqrt(x * x + y * y);
            GKernel[x + 2][y + 2] = (exp(-(r * r) / s)) / (M_PI * s);
            sum += GKernel[x + 2][y + 2];
        }
    }
  
    // normalising the Kernel
    for (int i = 0; i < kernelsize; ++i)
        for (int j = 0; j < kernelsize; ++j)
            GKernel[i][j] /= sum;
}
  
}

int main(int argc, char** argv)
{
    cv::namedWindow("Original Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Processed Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);

    cv::Mat h_img = cv::imread(argv[1]);
    cv::cuda::GpuMat d_img, d_result, d_gaussMat;
    cv::Mat_<cv::Vec3b> h_result;



    d_img.upload(h_img);
    d_result.upload(h_img);

    cv::imshow("Original Image", d_img);
    int frameSize = atoi(argv[2]);
    psrs::AddFrame(h_img, h_result, frameSize);
    double GKernel[5][5];
    psrs::FilterCreation(GKernel, 5);
    
    cv::Mat h_gaussMat(5, 5, CV_16S, GKernel);
    //for (int i = 0; i < 5; ++i) {
    //  for (int j = 0; j < 5; ++j)
    //      cout << GKernel[i][j] << "\t";
    //   cout << endl;
    //}
    cout << 'M ='  << h_gaussMat << endl << endl;
    d_gaussMat.upload(h_gaussMat);

    auto begin = chrono::high_resolution_clock::now();
    const int iter = 10000;

        
    for (int i = 0; i < iter; i++)
    {
        startCUDA(d_img, d_result,d_gaussMat);
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
