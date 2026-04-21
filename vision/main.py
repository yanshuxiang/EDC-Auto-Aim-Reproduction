import cv2
import numpy as np
from vision.src.laser import Laser
from vision.src.target import Target
from vision.src.capture import FrameCapture
from vision.src.saver import ResultSaver
from vision.src.communication import SerialCommunicator
from vision.src.filter import KalmanFilterTracker
import os
import shutil
import time

# 新增：控制是否录制的变量
record_video = False  # 设置为True时录制，False时不录制

debug = True

def init_dir():
    debug_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "debug"))
    if os.path.isdir(debug_dir):
        for name in os.listdir(debug_dir):
            path = os.path.join(debug_dir, name)
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)

# 初始化 debug 目录
init_dir()

target=Target(isdebug=debug)
laser=Laser(isdebug=debug)
capture=FrameCapture(40, width=640, height=480, fps=120)
saver=ResultSaver()
serial=SerialCommunicator()
tracker=KalmanFilterTracker(dt=1/120, lost_threshold=10)
frame_count = 0


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
        
        # 使用卡尔曼滤波处理目标位置，以平滑抖动并在短暂丢失时预测
        filtered_pos, is_tracking = tracker.update(tar_pos)
        
        # 发送数据到串口 (使用滤波后的坐标和追踪标志)
        serial.send_data(filtered_pos, las_pos, is_found=is_tracking)
        
        # 将每一帧保存到 debug/frames 文件夹
        if debug:
            frames_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "debug", "frames"))
            if not os.path.exists(frames_dir):
                os.makedirs(frames_dir, exist_ok=True)
            frame_path = os.path.join(frames_dir, f"frame_{frame_count:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_count += 1
        
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



