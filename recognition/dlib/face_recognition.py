import cv2
import face_recognition


class FaceRecognition:
    def __init__(self):
        pass

    def face_encodings(self, face, face_locations):
        rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # face recognition module needs (top, right, bottom, left) ordering
        known_face_locations = [
            (start_y, end_x, end_y, start_x)
            for (start_x, start_y, end_x, end_y) in face_locations
        ]
        return face_recognition.face_encodings(rgb_face, known_face_locations)
