#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

void unsharpMask(const Mat& src, Mat& dst, float sigma, int thresh, float scale)
{
    CV_Assert( src.type() == CV_8UC3 );

    Mat smoothed;
    GaussianBlur(src, smoothed, Size(cvRound(sigma*3)|1, cvRound(sigma*3)|1), sigma, sigma);
    dst.create(src.size(), src.type());

    for( int i = 0; i < src.rows; i++ )
    {
        for( int j = 0; j < src.cols*3; j++ )
        {
            uchar pix = src.at<uchar>(i, j);
            uchar spix = smoothed.at<uchar>(i, j);
            int diff = pix - spix;
            float result = abs(diff) < thresh ? (float)pix : pix + scale*diff;
            dst.at<uchar>(i, j) = saturate_cast<uchar>(result);
        }
    }
}

Mat img;
int sigma = 3;

void on_track(int sigma, void*)
{
    Mat result;
    unsharpMask(img, result, sigma, 10, 2);
    imshow("sharpen", result);
}

int main(int argc, char** argv)
{
    const char* filename = argc > 1 ? argv[1] : "nn.jpg";
    img = imread(filename);
    if( img.empty() ) { printf("error: cannot open image %s\n", filename); return 0; }
    namedWindow("sharpen");
    createTrackbar("sigma", "sharpen", &sigma, 11, on_track, 0);
    on_track(sigma, 0);

    waitKey(0);
    return 0;
}
