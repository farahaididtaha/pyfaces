import os
from pathlib import Path

import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array


class EmotionDetector:
    EMOTION = {
        "Angry": 0,
        "Disgust": 1,
        "Fear": 2,
        "Happy": 3,
        "Neutral": 4,
        "Sad": 5,
        "Surprise": 6,
    }

    def __init__(self):
        base_path = Path(__file__).parent.resolve()
        self.detector = load_model(os.path.join(base_path, "model_v6_23.hdf5"))

    def detect(self, face):
        # TODO: check if input image is multiple channel or single channel
        roi = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
        roi = cv2.resize(roi, (48, 48), interpolation=cv2.INTER_AREA)
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        predicted_class = np.argmax(self.detector.predict(roi))
        label_map = dict((v, k) for k, v in self.EMOTION.items())
        predicted_label = label_map[predicted_class]
        print(predicted_label)


