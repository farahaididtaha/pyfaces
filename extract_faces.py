import os
from pathlib import Path
import cv2
from detectors.face_detector.opencv_dnn.face_detector import FaceDetector
from utils.paths import list_images
from utils.image import resize as image_resize


if __name__ == '__main__':
    detector = FaceDetector()
    previous_name = None
    index = 0
    image_paths = list(list_images("data/original"))

    for image_path in image_paths:
        print(f"Processing {image_path}")
        image = cv2.imread(image_path)
        output_dir = Path(image_path).parent.parent.parent.resolve()
        name = image_path.split(os.path.sep)[-2]
        if previous_name == name:
            index += 1
        else:
            index = 0
        previous_name = name
        output_person_dir = os.path.join(output_dir, "output", name)

        face_rects, _ = detector.detect(image)
        for face_rect in face_rects:
            start_x, start_y, end_x, end_y = face_rect
            face = image[start_y:end_y, start_x:end_x]

            if not os.path.exists(output_person_dir):
                os.mkdir(output_person_dir)

            new_file_path = os.path.join(output_person_dir, f"{index:03d}.jpg")
            cv2.imwrite(new_file_path, face)
