import numpy as np
import cvlib


class GenderDetector:
    def __init__(self, enable_gpu=True):
        self.detector = cvlib.gender_detection.GenderDetection()
        self.enable_gpu = enable_gpu

    def detect(self, face):
        genders, confidences = self.detector.detect_gender(face, self.enable_gpu)
        idx = np.argmax(confidences)
        return genders[idx], confidences[idx]
