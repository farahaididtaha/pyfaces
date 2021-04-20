from enum import Enum


class FaceDetectorAlgorithm(Enum):
    # They are super fast, even on resource-constrained devices
    # They are lightweight, The Haar cascade model size is tiny (930 KB)
    # But it is prone to false-positive detections
    HAAR_CASCADE = 0

    # OpenCV’s deep learning face detector
    # based on the Single Shot Detector (SSD) framework with a ResNet base network
    # I recommend this method
    OPENCV_DNN = 1

    # HOG(Histogram of Gradient) + Linear SVM face detector
    # that is accurate and computationally efficient.
    # But it is not invariant to changes in rotation and viewing angle.
    DLIB_HOG = 2

    # CNN based face detector trained on 3 million faces
    # But it is computationally expensive if you don't have GPU
    DLIB_DNN = 3

    # The cutting-edge face detector,
    RETINA_FACE = 4
