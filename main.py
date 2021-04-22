import argparse
import pickle
import time
import cv2
from detectors.face_detector.opencv_dnn.face_detector import FaceDetector
from detectors.gender_detector.detector import GenderDetector
from detectors.emotion_detector.detector import EmotionDetector
from recognition.dlib.face_recognition import FaceRecognition
from utils.image import resize
from utils.video import FileVideoStream


face_recognizer = pickle.loads(open("models/recognizer.pickle", "rb").read())
le = pickle.loads(open("models/le.pickle", "rb").read())
face_detector = FaceDetector()
gender_detector = GenderDetector()
emotion_detector = EmotionDetector()
face_recognition = FaceRecognition()


def frame_recognize(frame, wait=0):
    t1 = time.time()
    faces, face_confidences = face_detector.detect(frame)
    if len(faces) < 1:
        return
    t2 = time.time()
    elapsed = f"{(t2 - t1):.2f}"
    encodings = face_recognition.face_encodings(frame, faces)
    pred_probs = face_recognizer.predict_proba(encodings)

    for face_rect, pred_prob in zip(faces, pred_probs):
        start_x, start_y, end_x, end_y = face_rect
        face = frame[start_y:end_y, start_x:end_x]
        print(face_rect)
        gender, gender_confidence = gender_detector.detect(face)
        emotion = emotion_detector.detect(face)
        pred = pred_prob.argmax()
        prob = pred_prob[pred]

        if prob < 0.7:
            continue

        name = le.classes_[pred]
        cv2.rectangle(img=frame, pt1=(start_x, start_y), pt2=(end_x, end_y),
                      color=(0, 255, 0), thickness=1)
        cv2.putText(img=frame,
                    text=f"{name}: {gender} & {emotion}",
                    org=(start_x, start_y - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.4, thickness=1, color=(0, 255, 0))

    # cv2.putText(img=frame, text=f"Took {elapsed}s ", org=(10, 15),
    #             fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
    #             thickness=1, color=(0, 255, 0))
    cv2.imshow("Faces", frame)
    return cv2.waitKey(wait) & 0xFF


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", type=str, help="path to the image")
    ap.add_argument("-v", "--video", type=str, help="path to the video")
    args = ap.parse_args()

    if args.video:
        fvs = FileVideoStream(args.video).start()
        time.sleep(3)
        while fvs.more():
            frame = fvs.read()
            frame = resize(frame, width=600)
            key = frame_recognize(frame, wait=1)
            if key == ord("q"):
                break

    elif args.image:
        image = cv2.imread(args.image)
        frame_recognize(image)
