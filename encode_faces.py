import argparse
import os
import pickle
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from utils.paths import list_images
from recognition.dlib.face_recognition import FaceRecognition
# from detectors.face_detector.opencv_dnn.face_detector import FaceDetector


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--directory", help="directory of faces", default="data/output")
    args = ap.parse_args()

    face_recognition = FaceRecognition()

    names = []
    face_encodings = []
    # TODO: Make this generator as well
    for image_path in list_images(args.directory):
        name = image_path.split(os.path.sep)[-2]
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        face_regions = [(0, 0, w, h)]
        encodings = face_recognition.face_encodings(image, face_regions)
        for encoding in encodings:
            names.append(name)
            face_encodings.append(encoding)

    data = {"encodings": face_encodings, "names": names}
    with open("models/encodings.pickle", "wb") as f:
        f.write(pickle.dumps(data))
