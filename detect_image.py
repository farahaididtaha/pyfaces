import argparse
import cv2
import time

from detectors.face_detector.haar_cascade.face_detector import FaceDetector as HaarCascadeFaceDetector
from detectors.face_detector.opencv_dnn.face_detector import FaceDetector as OpenCVDNNFaceDetector
from detectors.face_detector.dlib_hog.face_detector import FaceDetector as DlibHogFaceDetector
from detectors.face_detector.dlib_cnn.face_detector import FaceDetector as DlibCnnFaceDetector
from detectors.gender_detector.detector import GenderDetector


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--method", type=str, help="specify face detector algorithm")
    ap.add_argument("-i", "--image", type=str, help="path to the image")
    args = ap.parse_args()

    face_detectors = {
        "haar": HaarCascadeFaceDetector,
        "dnn": OpenCVDNNFaceDetector,
        "hog": DlibHogFaceDetector,
        "cnn": DlibCnnFaceDetector,
    }
    try:
        face_detector_class = face_detectors[args.method]
    except KeyError:
        pass
    else:
        t1 = time.time()
        gender_detector = GenderDetector()
        face_detector = face_detector_class()
        image = cv2.imread(args.image)
        faces, face_confidences = face_detector.detect(image)
        t2 = time.time()
        elapsed = f"{(t2 - t1):.2f}"
        for face, face_confidence in zip(faces, face_confidences):
            start_x, start_y, end_x, end_y = face
            gender, gender_confidence = gender_detector.detect(image[start_y:end_y, start_x:end_x])
            cv2.rectangle(img=image, pt1=(start_x, start_y), pt2=(end_x, end_y),
                          color=(0, 255, 0), thickness=1)
            cv2.putText(img=image,
                        text=f"Face {face_confidence * 100:.2f}% & {gender}: {gender_confidence * 100:.2f}%",
                        org=(start_y, start_y - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, thickness=1, color=(0, 255, 0))

        cv2.putText(img=image, text=f"Took {elapsed}s ", org=(10, 15),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                    thickness=1, color=(0, 255, 0))
        cv2.imshow("Faces", image)
        cv2.waitKey(0)
