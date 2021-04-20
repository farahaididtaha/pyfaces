import os
from pathlib import Path

import cv2
import dlib


class FaceDetector:
    """
    HOG + Linear SVM face detector
    HOG stands for Histogram of Gradients
    """
    def __init__(self):
        base_path = Path(__file__).parent.resolve()
        self.detector = dlib.cnn_face_detection_model_v1(
            os.path.join(base_path, "mmod_human_face_detector.dat")
        )
        # Number of times to upsample an image before applying face detection.
        # To detect small faces in a large input image, we may wish to increase
        # the resolution of the input image, thereby making the smaller faces appear larger.
        # Doing so allows our sliding window to detect the face.
        self.upsample = 1

    def detect(self, image):
        # Convert the image from BGR to RGB channel ordering (dlib expects RGB images)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = self.detector(rgb, self.upsample)
        faces = [
            (
                max(0, face.rect.left()),
                max(0, face.rect.top()),
                min(face.rect.right(), image.shape[1]),
                min(face.rect.bottom(), image.shape[0])
            )
            for face in faces
        ]
        return faces, [1] * len(faces)
