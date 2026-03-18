import os
import shutil

import cv2
import numpy as np


class ResultSaver:
    def __init__(
        self,
        output_video_path="output.mp4",
        wrong_dir="\debug\wrong_pic",
        wrong_pure_dir="debug\wrong_pic_pure",
        normal_dir=r"debug\normal",
        fps=30.0,
        frame_size=(1280, 720),
    ):
        self.output_video_path = output_video_path
        self.wrong_dir = wrong_dir
        self.wrong_pure_dir = wrong_pure_dir
        self.normal_dir = normal_dir
        self.fps = fps
        self.frame_size = frame_size
        self.wrong_count = 0
        self.save_count=0

        self._prepare_dirs()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(self.output_video_path, fourcc, self.fps, self.frame_size)

    def _prepare_dirs(self):
        for target_dir in (self.wrong_dir, self.wrong_pure_dir):
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            os.makedirs(target_dir)

    def save_wrong_frame(self, frame, binary_frame, contours):
        self.wrong_count += 1
        marked_frame = frame.copy()
        cv2.drawContours(marked_frame, contours, -1, (0, 0, 255), 3)
        wrong_view = np.vstack((marked_frame, cv2.cvtColor(binary_frame, cv2.COLOR_GRAY2BGR)))

        wrong_path = os.path.join(self.wrong_dir, f"wrong{self.wrong_count}.jpg")
        wrong_pure_path = os.path.join(self.wrong_pure_dir, f"wrong_frame{self.wrong_count}.jpg")
        cv2.imwrite(wrong_path, wrong_view)
        cv2.imwrite(wrong_pure_path, frame)

    def save_frame(self,frame):
        self.save_count += 1
        path=os.path.join(self.normal_dir, self.save_count)
        cv2.imwrite(path, frame)

    def write_render_frame(self, frame):
        self.writer.write(frame)

    def release(self):
        if self.writer is not None:
            self.writer.release()
