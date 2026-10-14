"""Bounded preview preparation. Decoder creation/read/release all stay on one thread."""
import queue
import threading

import cv2


class PreviewReader:
    def __init__(self, path, encode, start_frame=0):
        self.path, self.encode = str(path), encode
        self.start_frame = start_frame
        self.queue = queue.Queue(maxsize=4)
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _put(self, item):
        while not self.stopped.is_set():
            try:
                self.queue.put(item, timeout=0.05)
                return
            except queue.Full:
                pass

    def _run(self):
        cap = cv2.VideoCapture(self.path)
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if self.start_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            while not self.stopped.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                self._put((self.encode(frame), fps))
        except Exception as error:
            self._put(error)
        finally:
            cap.release()
            self._put(None)

    def read(self):
        item = self.queue.get(timeout=1)
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        self.stopped.set()
        self.thread.join(timeout=2)
