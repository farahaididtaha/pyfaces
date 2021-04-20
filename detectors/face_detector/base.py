from dataclasses import dataclass


@dataclass
class BoundingBox:
    left: int
    top: int
    width: int
    height: int


class BaseFaceDetector:
    def detect(self):
        raise NotImplementedError
