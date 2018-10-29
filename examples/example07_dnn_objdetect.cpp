#include "opencv2/opencv.hpp"
#include "opencv2/bioinspired.hpp"

using namespace cv;
using namespace std;

int main(int, char** argv)
{
    const int size0 = 300;
    static const char* classNames[] =
    {
        "background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant",
        "sheep", "sofa", "train", "tvmonitor"
    };
    int maxlabel = (int)(sizeof(classNames)/sizeof(classNames[0]));

    dnn::Net net = dnn::readNetFromCaffe("MobileNetSSD_deploy_generated.prototxt",
                                  "MobileNetSSD_deploy_generated.caffemodel");
    if(net.empty())
    {
        printf("error: cannot read MobileNet SSD network\n");
        return 0;
    }

    VideoCapture cap(0);
    if( !cap.isOpened())
    {
        printf("error: cannot initialize video capture\n");
        return 0;
    }

    Mat frame, frame_r, blob;

    for(;;) {
        cap >> frame; if(frame.empty()) break;
        resize(frame, frame_r, Size(size0, size0));
        dnn::blobFromImage(frame_r, blob, 1./127.5, Size(size0, size0), 127.5);

        net.setInput(blob);
        Mat detections = net.forward();
        CV_Assert( detections.dims == 4 && (detections.size[2] == 0 || detections.size[3] == 7));
        for( int i = 0; i < detections.size[2]; i++ )
        {
            int idx[] = {0, 0, i, 0};
            float* row = detections.ptr<float>(idx);
            float confidence = row[2];
            if( confidence < 0.2f )
                continue;
            int class_id = cvRound(row[1]);
            if( class_id < 0 || class_id >= maxlabel )
                continue;
            int x0 = cvRound(row[3] * frame.cols);
            int y0 = cvRound(row[4] * frame.rows);
            int x1 = cvRound(row[5] * frame.cols);
            int y1 = cvRound(row[6] * frame.rows);
            rectangle(frame, Point(x0, y0), Point(x1, y1), Scalar(100, 255, 100));
            string label = format("%s: %.2f", classNames[class_id], confidence);
            int bl = 0;
            Size lsize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &bl);
            rectangle(frame, Point(x0,y0), Point(x0+lsize.width, y0+lsize.height+bl), Scalar(100,255,100), -1);
            putText(frame, label, Point(x0, y0 + lsize.height), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1, LINE_AA);
        }
        imshow("detections", frame);
        if( waitKey(30) >= 0 ) break;
    }
    return 0;
}
