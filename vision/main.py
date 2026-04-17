import cv2
import numpy as np
from vision.src.laser import Laser
from vision.src.target import Target
from vision.src.capture import FrameCapture
from vision.src.saver import ResultSaver
from vision.src.communication import SerialCommunicator
import os
import shutil
import time

# 新增：控制是否录制的变量
record_video = False  # 设置为True时录制，False时不录制

debug=False

def init_dir():
    debug_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "debug"))
    if os.path.isdir(debug_dir):
        for name in os.listdir(debug_dir):
            path = os.path.join(debug_dir, name)
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)

target=Target(isdebug=debug)
laser=Laser(isdebug=debug)
capture=FrameCapture(40, width=640, height=480, fps=120)
saver=ResultSaver()
serial=SerialCommunicator()


# 修改：只在需要录制时才创建recorder
if record_video:
    recorder = saver.create_recorder(capture.get_size(), capture.get_fps())
else:
    recorder = None

fps=0
t=time.time()

try:
    while True:
        fps+=1
        t2=time.time()
        if(t2-t>1):
            print("fps:", fps)
            fps=0
            t=time.time()
        ret, frame = capture.read()
        if not ret:
            break
        # print("running")
        # 处理帧...
        tar_pos=target.detect(frame)
        las_pos=laser.detect(frame)
        
        # 修改：只在需要录制时才写入视频
        if record_video and recorder is not None:
            try:
                recorder.write(frame)
            except Exception as e:
                print(f"Error writing to video: {e}")
                # 如果写入失败，停止录制
                record_video = False
                if recorder is not None:
                    recorder.release()
                    recorder = None

except KeyboardInterrupt:
    print("Recording stopped by user.")



