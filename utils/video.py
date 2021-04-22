from threading import Thread
import cv2
import time
from queue import Queue


class FileVideoStream:
    def __init__(self, path, transform=None, queue_size=128):
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        self.transform = transform
        self.Q = Queue(maxsize=queue_size)
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True

    def start(self):
        self.thread.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                break

            if not self.Q.full():
                (grabbed, frame) = self.stream.read()

                if not grabbed:
                    self.stopped = True
                if self.transform:
                    frame = self.transform(frame)

                self.Q.put(frame)
            else:
                time.sleep(0.1)  # Rest for 10ms, we have a full queue

        self.stream.release()

    def read(self):
        return self.Q.get()

    def running(self):
        return self.more() or not self.stopped

    def more(self):
        tries = 0
        while self.Q.qsize() == 0 and not self.stopped and tries < 5:
            time.sleep(0.1)
            tries += 1

        return self.Q.qsize() > 0

    def stop(self):
        self.stopped = True
        self.thread.join()
