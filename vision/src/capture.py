import cv2


class FrameCapture:
    def __init__(self, source, width=1920, height=1080, fps=None):
        self.source = source
        self.req_width = width
        self.req_height = height
        self.req_fps = fps

        self.cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open source: {source}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.req_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.req_height)

        if self.req_fps is not None:
            self.cap.set(cv2.CAP_PROP_FPS, self.req_fps)

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 1e-6:
            self.fps = 30.0

        print(f"[FrameCapture] source={source}")
        print(f"[FrameCapture] requested size=({self.req_width}, {self.req_height})")
        print(f"[FrameCapture] actual size=({self.width}, {self.height})")
        print(f"[FrameCapture] fps={self.fps}")

    def read(self):
        return self.cap.read()

    def get_size(self):
        return (self.width, self.height)

    def get_fps(self):
        return self.fps

    def release(self):
        if self.cap is not None:
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()