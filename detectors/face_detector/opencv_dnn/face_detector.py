import os
from pathlib import Path

import cvlib
import numpy as np
import cv2


class FaceDetector:
    """
    OpenCV’s deep learning face detector is based on the
    Single Shot Detector (SSD) framework with a ResNet base network
    """
    def __init__(self):
        # cvlib interally use opencv dnn model
        self.use_cvlib = True
        self.threshold = 0.5

        # but if we don't want to 3rd party library, we can use opencv dnn module directly
        self.detector = None

    def load_caffe_model(self):
        base_path = Path(__file__).parent.resolve()
        # .prototxt file define the model architecture (i.e., the layers themselves)
        # .caffemodel contains the weights for the actual layers
        self.detector = cv2.dnn.readNetFromCaffe(
            prototxt=os.path.join(base_path, "deploy.prototxt.txt"),
            caffeModel=os.path.join(base_path, "res10_300x300_ssd_iter_140000.caffemodel"),
        )

    def update_parameter(self, threshold=0.5, use_cvlib=True):
        self.threshold = threshold
        self.use_cvlib = use_cvlib
        if not self.use_cvlib and self.net is None:
            self.load_caffe_model()

    def detect(self, image):
        if self.use_cvlib:
            faces, confidences = cvlib.detect_face(image)
        else:
            # TODO: Make constant image size, and mean value of rgb
            # (104.0, 177.0, 123.0) is the mean value of rgb, this can be use as normalization

            h, w = image.shape[:2]
            blob = cv2.dnn.blobFromImage(
                image=cv2.resize(image, (300, 300)),
                scalefactor=1,
                size=(300, 300),
                mean=(104.0, 177.0, 123.0)
            )
            self.detector.setInput(blob)
            detections = self.detector.forward()

            faces = []
            confidences = []

            for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with the prediction
                confidence = detections[0, 0, i, 2]

                # ignore detections with low confidence
                if confidence < self.threshold:
                    continue

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (start_x, start_y, end_x, end_y) = box.astype("int")
                faces.append([start_x, start_y, end_x, end_y])
                confidences.append(confidence)

        return faces, confidences
