import cv2


class FrameCapture:
    def __init__(self, source, width=640, height=480, fps=None):
        self.source = source
        self.req_width = width
        self.req_height = height
        self.req_fps = fps

        self.cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open source: {source}")

        # 方法二：使用 MJPG 四字符编码（关键优化）
        # 设置 MJPG 格式可以大幅降低 USB 带宽需求，支持更高帧率
        mjpg_fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        ret_fourcc = self.cap.set(cv2.CAP_PROP_FOURCC, mjpg_fourcc)
        if ret_fourcc:
            print(f"[FrameCapture] FOURCC 设置为 MJPG 成功")
        else:
            print(f"[FrameCapture] 警告：FOURCC 设置可能未生效")

        # 设置分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.req_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.req_height)

        # 设置帧率
        if self.req_fps is not None:
            self.cap.set(cv2.CAP_PROP_FPS, self.req_fps)

        # 读取实际配置
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # 获取实际 FOURCC 格式
        actual_fourcc = self.cap.get(cv2.CAP_PROP_FOURCC)
        if actual_fourcc > 0:
            fourcc_chars = chr(int(actual_fourcc) & 0xFF) + \
                          chr((int(actual_fourcc) >> 8) & 0xFF) + \
                          chr((int(actual_fourcc) >> 16) & 0xFF) + \
                          chr((int(actual_fourcc) >> 24) & 0xFF)
        else:
            fourcc_chars = "Unknown"
        
        if self.fps <= 1e-6:
            self.fps = 30.0

        print(f"[FrameCapture] source={source}")
        print(f"[FrameCapture] requested size=({self.req_width}, {self.req_height})")
        print(f"[FrameCapture] actual size=({self.width}, {self.height})")
        print(f"[FrameCapture] FOURCC={fourcc_chars}")
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
