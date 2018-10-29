#include "opencv2/opencv.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/bioinspired.hpp"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    ocl::setUseOpenCL(false);
    VideoCapture cap;
    if( argc > 1 )
        cap.open(argv[1]);
    else
        cap.open(0);
    Ptr<bioinspired::Retina> retina;
    Mat frame0, frame, parvo, magno;
    for(;;) {
        cap >> frame0; if(frame0.empty()) break;
        if( frame0.cols > 640 )
        {
            double sf = 640./frame0.cols;
            resize(frame0, frame, Size(), sf, sf, INTER_LINEAR);
        }
        else
            frame = frame0;
        if(!retina)
            retina = bioinspired::Retina::create(frame.size());
        retina->run(frame);
        retina->getParvo(parvo);
        retina->getMagno(magno);
        imshow("Parvo", parvo);
        imshow("Magno", magno);
        if(waitKey(30) != -1) break;
    }
    return 0;
}
