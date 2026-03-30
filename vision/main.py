import cv2
import numpy as np
from vision.src.laser import Laser
from vision.src.target import Target
from vision.src.capture import FrameCapture
import os
import shutil

isdebug=False

def init_dir():
    debug_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "debug"))
    if os.path.isdir(debug_dir):
        for name in os.listdir(debug_dir):
            path = os.path.join(debug_dir, name)
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)

target=Target()
laser=Laser()

with FrameCapture('../media/1.mp4') as capture:
    while True:
        ret, frame = capture.read()
        if not ret:
            break

        if isdebug:
            init_dir()

        target_pos=Target.detect(frame)
        laser_pos=Laser.detect(frame)
        pass




