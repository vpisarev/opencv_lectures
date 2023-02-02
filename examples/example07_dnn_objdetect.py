import sys, numpy as np, cv2 as cv

size0 = 300
classNames = ('background', 'aeroplane', 'bicycle', 'bird', 'boat',
              'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant',
              'sheep', 'sofa', 'train', 'tvmonitor')
net = cv.dnn.readNetFromCaffe('MobileNetSSD_deploy_generated.prototxt',
                              'MobileNetSSD_deploy_generated.caffemodel')
cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret: break
    blob = cv.dnn.blobFromImage(frame, 1./127.5, (size0, size0), 127.5)
    net.setInput(blob)
    detections = net.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:
            class_id = int(detections[0, 0, i, 1])
            x0 = int(detections[0, 0, i, 3] * frame.shape[1])
            y0 = int(detections[0, 0, i, 4] * frame.shape[0])
            x1 = int(detections[0, 0, i, 5] * frame.shape[1])
            y1 = int(detections[0, 0, i, 6] * frame.shape[0])
            cv.rectangle(frame, (x0, y0), (x1, y1), (100, 255, 100), 2)
            label = classNames[class_id] + ": " + str(round(confidence, 2))
            lsize, bl = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv.rectangle(frame, (x0, y0), (x0 + lsize[0], y0 + lsize[1] + bl),
                      (100, 255, 100), cv.FILLED)
            cv.putText(frame, label, (x0, y0 + lsize[1]),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
    cv.imshow("detections", frame)
    if cv.waitKey(30) >= 0: break
