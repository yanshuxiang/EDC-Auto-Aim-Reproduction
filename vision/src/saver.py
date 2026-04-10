from pathlib import Path

import cv2


class ResultSaver:
    """
    负责将视频保存到 `debug` 下的指定子目录（默认 `output`）。
    """

    def __init__(self, name="output.mp4", base_dir=None, fps=30.0, frame_size=None, output_subdir="output"):
        self.base_dir = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent.parent
        self.output_dir = (self.base_dir / "debug" / output_subdir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        filename = Path(str(name)).name
        if not filename:
            filename = "output.mp4"
        if not filename.lower().endswith(".mp4"):
            filename = f"{filename}.mp4"

        self.output_path = str((self.output_dir / filename).resolve())
        self.fps = fps
        self.frame_size = frame_size
        self.writer = None

    def _open_writer(self, frame):
        if self.writer is not None:
            return

        if self.frame_size is None:
            height, width = frame.shape[:2]
            self.frame_size = (width, height)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, self.frame_size)
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter: {self.output_path}")

    def write(self, frame):
        if frame is None:
            return

        self._open_writer(frame)
        target_size = self.frame_size
        current_size = (frame.shape[1], frame.shape[0])
        if current_size != target_size:
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        self.writer.write(frame)

    def write_render_frame(self, frame):
        self.write(frame)

    def release(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False
