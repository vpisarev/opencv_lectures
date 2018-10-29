#include "opencv2/opencv.hpp"
using namespace cv;
int main(int argc, char** argv) {
    Mat img, gray, edges;
    int smoothness = 15;
    bool firstframe = true;
    VideoCapture cap(0); // для чтения из видеофайла заменить на VideoCapture cap(argv[1])
    for(;;) {
        cap >> img; if(img.empty()) break;
        cvtColor(img, gray, COLOR_BGR2GRAY);
        double sigma = smoothness*0.1 + 1;
        int ksize = cvRound(sigma*4+1)|1;
        GaussianBlur(gray, gray, Size(ksize, ksize), sigma);
        Canny(gray, edges, 0, 50);
        imshow("edges", edges);
        if(firstframe) {
            createTrackbar("smoothness", "edges", &smoothness, 30, 0, 0);
            firstframe = false;
        }
        if(waitKey(30) >= 0) break;
    }
    return 0;
}
