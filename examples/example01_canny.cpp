#include "opencv2/opencv.hpp"
using namespace cv;
int main(int argc, char** argv)
{
    Mat img, gray, edges;
    img = imread(argc > 1 ? argv[1] : "lena.jpg", 1);
    if( img.empty() )
    {
        printf("error: could not read the image\n");
        return 0;
    }
    imshow("original", img);
    cvtColor(img, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, gray, Size(7, 7), 1.5);
    Canny(gray, edges, 0, 50);
    imshow("edges", edges);
    imwrite("result.png", edges);
    waitKey();
    return 0;
}
