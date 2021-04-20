import os
from pathlib import Path

import cv2


class FaceDetector:
    def __init__(self):

        self.scale_factor = 1.1
        # This value is used to create the scale image pyramid.
        # This value indicates how much the image size is reduced at each image scale.
        # A value of 1.1 indicates that we are reducing the size of the image by 10% at each level in the pyramid.

        self.min_neighbors = 5
        # This value indicates how many neighbors each window should have for the area
        # in the window to be considered a face.
        # The cascade classifier will detect multiple windows around a face.
        # This parameter controls how many rectangles (neighbors) need to be detected
        # for the window to be labeled a face.

        self.min_size = (30, 30)
        base_path = Path(__file__).parent.resolve()
        self.detector = cv2.CascadeClassifier(
            os.path.join(base_path, "haarcascade_frontalface_default.xml")
        )

    def update_parameter(self, scale_factor, min_neighbors, min_size):
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def detect(self, image):
        # (x, y, w, h)
        faces = self.detector.detectMultiScale(
            image=image,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        # INFO: Cascade does not provide confidence
        faces = [
            (face[0], face[1], face[0] + face[2], face[1] + face[3])
            for face in faces
        ]
        return faces, [1] * len(faces)
