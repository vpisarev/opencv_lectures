#include "opencv2/opencv.hpp"
using namespace cv;
Mat img, gray, edges;
int smoothness = 15;
static void on_smoothness(int, void*) {
    cvtColor(img, gray, COLOR_BGR2GRAY);
    double sigma = smoothness*0.1 + 1;
    int ksize = cvRound(sigma*4+1)|1;
    GaussianBlur(gray, gray, Size(ksize, ksize), sigma);
    Canny(gray, edges, 0, 50);
    imshow("edges", edges);
}
int main(int argc, char** argv) {
    img = imread(argc > 1 ? argv[1] : "lena.jpg", 1);
    imshow("original", img); imshow("edges", img);
    createTrackbar("smoothness", "edges", &smoothness, 30, on_smoothness, 0);
    on_smoothness(0, 0);
    waitKey(); return 0;
}
