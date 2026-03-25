import cv2


class FrameCapture:
    def __init__(self, source):
        """
        初始化视频帧采集器，并立即尝试打开输入源。

        参数说明：
        - source: OpenCV 可识别的视频源，可以是摄像头索引、视频文件路径或流地址。

        初始化后会缓存输入源的宽、高和帧率信息，供后续外部模块直接读取。
        如果底层 VideoCapture 打开失败，则抛出异常，避免系统在无效输入源上继续运行。
        """
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open source: {source}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 1e-6:
            self.fps = 60.0

    def read(self):
        """
        从当前输入源读取一帧数据。

        返回值直接沿用 OpenCV `VideoCapture.read()` 的约定：
        - 第一个值表示是否成功读取到帧。
        - 第二个值是读取到的图像；失败时通常为 None。
        """
        return self.cap.read()

    def get_size(self):
        """
        返回输入源的帧尺寸。

        返回格式为 `(width, height)`，便于下游模块初始化显示窗口、
        视频写入器或其他依赖固定分辨率的组件。
        """
        return (self.width, self.height)

    def get_fps(self):
        """
        返回输入源帧率。

        当底层设备或文件未能正确提供帧率信息时，构造函数中会回退到 60.0，
        因此这里返回的始终是一个可用的数值。
        """
        return self.fps

    def release(self):
        """
        释放底层 VideoCapture 资源。

        该方法应在采集结束后调用，以关闭摄像头/文件句柄并释放系统资源。
        """
        if self.cap is not None:
            self.cap.release()

    def __enter__(self):
        """
        支持 `with` 语句的上下文管理入口。

        返回当前实例本身，使调用方可以通过 `with FrameCapture(...) as cap:`
        的方式安全使用采集器。
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        支持 `with` 语句的上下文管理出口。

        无论上下文内部是否抛出异常，都会调用 `release()` 释放采集资源。

        参数说明：
        - exc_type: 异常类型；若上下文正常结束则为 None。
        - exc_value: 异常对象；若上下文正常结束则为 None。
        - traceback: 异常回溯对象；若上下文正常结束则为 None。
        """
        self.release()
