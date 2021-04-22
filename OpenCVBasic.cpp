#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>  // for high_resolution_clock

using namespace std;

cv::Vec3f rgb2xyz(const cv::Vec3f src ){
    cv::Vec3f tmp=cv::Vec3f(
        (src.r>.04045)?pow((src.r+.055)/1.055,2.4):src.r/12.92,
        (src.g>.04045)?pow((src.g+.055)/1.055,2.4):src.g/12.92,
        (src.b>.04045)?pow((src.b+.055)/1.055,2.4):src.b/12.92
    );
    cv::Vec3f mat=cv::Vec3f(
        .4124,.3576,.1805,
        .2126,.7152,.0722,
        .0193,.1192,.9505
    );
    return 100.0*(tmp*mat);
}
cv::Vec3f xyz2lab(const cv::Vec3f src){
    cv::Vec3f n=c/cv::Vec3f(95.047,100.,108.883),
         v=cv::Vec3f(
        (src.x>.008856)?pow(src.x,1./3.):(7.787*src.x)+(16./116.),
        (src.y>.008856)?pow(src.y,1./3.):(7.787*src.y)+(16./116.),
        (src.z>.008856)?pow(src.z,1./3.):(7.787*src.z)+(16./116.)
    );
    return vec3((116.*v.y)-16.,500.*(v.x-v.y),200.*(v.y-v.z));
}

cv::Vec3f rgb2lab(cv::Vec3f c) {
    vec3 lab = xyz2lab(rgb2xyz(c) );
    return vec3(lab.x / 100.0, 0.5 + 0.5 * (lab.y / 127.0), 0.5 + 0.5 * (lab.z / 127.0));
}

cv::Vec3f lab2xyz(cv::Vec3f src) {
    float fy = (src.x + 16.0) / 116.0;
    float fx = src.y / 500.0 + fy;
    float fz = fy - c.z / 200.0;
    return cv::Vec3f(
            95.047 * ((fx > 0.206897) ? fx * fx * fx : (fx - 16.0 / 116.0) / 7.787),
            100.000 * ((fy > 0.206897) ? fy * fy * fy : (fy - 16.0 / 116.0) / 7.787),
            108.883 * ((fz > 0.206897) ? fz * fz * fz : (fz - 16.0 / 116.0) / 7.787)
           );
  }
  cv::Vec3f xyz2rgb(cv::Vec3f src) {
    const mat3 mat = mat3(
              3.2406, -1.5372, -0.4986,
              -0.9689, 1.8758, 0.0415,
              0.0557, -0.2040, 1.0570
              );
    cv::Vec3f v = mat * (src/100.0);
    cv::Vec3f r;
    r.x = (v.r > 0.0031308) ? ((1.055 * pow(v.r, (1.0 / 2.4))) - 0.055) : 12.92 * v.r;
    r.y = (v.g > 0.0031308) ? ((1.055 * pow(v.g, (1.0 / 2.4))) - 0.055) : 12.92 * v.g;
    r.z = (v.b > 0.0031308) ? ((1.055 * pow(v.b, (1.0 / 2.4))) - 0.055) : 12.92 * v.b;
    return r;
  }

  cv::Vec3f lab2rgb(const cv::Vec3f src) {
    return xyz2rgb(lab2xyz(vec3(100.0 * src.x, 2.0 * 127.0 * (src.y - 0.5), 2.0 * 127.0 * (src.z - 0.5))));
  }
 

int main( int argc, char** argv )
{
  cv::Mat_<uchar> source = cv::imread ( argv[1], cv::IMREAD_GRAYSCALE);
  cv::Mat_<int> dst;
  cv::namedWindow("Source Image", cv::WINDOW_AUTOSIZE);
  cv::imshow("Source Image", source );

  auto begin = chrono::high_resolution_clock::now();
  const int iter = 1000;

  float PI = 3.14159265358979323846;

  dst.create(src.rows, src.cols);
  dst = cv::Vec3b(0, 0, 0);
  float shiftAngle=40;

  for (int i = 0; i < src.rows; i++)
    {
      for (int j = 0; j < src.cols; j++)
      {
        for (int c = 0; c < 3; c++)
        {
          vec3 lab(i,j)[c]= rgb2lab((cv::Vec3f)src.at<cv::Vec3b>(i, j)[c]);
          float C = sqrt(lab.y(i,j) + lab.z(i,j));
          loat h = atan2(lab.z(i,j)/ lab.y(i,j));
          h += (shiftAngle*PI)/180.0;
          vec2 ab = vec2(cos(h)*C, sin(h)*C);  
          lab.yz(i,j) = ab;
          dst(i, j) = lab2rgb((cv::Vec3f)lab.at<cv::Vec3b>(i, j));
				
        }
    }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end-begin;

  cout << "Total time: " << diff.count() << " s" << endl;
  cout << "Time for 1 iteration: " << diff.count()/iter << " s" << endl;
  cout << "IPS: " << iter/diff.count() << endl;
    
  cv::waitKey(0);
  return 0;
}

