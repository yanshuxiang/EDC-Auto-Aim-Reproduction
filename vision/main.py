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

# Initialize Serial Communicator (for Ubuntu default)
# 初始化串口通讯 (适应 Ubuntu 默认路径)
comm = SerialCommunicator(port='/dev/ttyUSB0', baudrate=115200)

t=time.time()
fps=0
camera_source = os.getenv("CAMERA_SOURCE", "22")
try:
    camera_source = int(camera_source)
except ValueError:
    pass

record_name = f"record_{time.strftime('%Y%m%d_%H%M%S')}.mp4"

try:
    with FrameCapture(camera_source) as capture, ResultSaver(
        name=record_name,
        fps=capture.get_fps(),
        output_subdir="records",
    ) as recorder:
        print(f"Recording camera[{camera_source}] to: {recorder.output_path}")
        while True:
            fps += 1
            t1 = time.time()
            if t1 - t > 1:
                print(fps)
                t = time.time()
                fps = 0

            ret, frame = capture.read()
            if not ret:
                break

            recorder.write(frame)

            if debug:
                init_dir()

            target_pos = target.detect(frame)
            laser_pos = laser.detect(frame)
            
            # Send data logic: send current detected positions to MCU
            # 数据发送逻辑：将当前检测到的位置发送给下位机
            if target_pos is not None and laser_pos is not None:
                # Send target and laser positions via serial
                # 通过串口发送目标和激光位置
                comm.send_data(target_pos, laser_pos)
                
                if debug:
                    print(f"Target: {target_pos}, Laser: {laser_pos}")

                # Visualization (Optional)
                cv2.circle(frame, target_pos, 5, (0, 255, 0), -1)  # Target in Green
                cv2.circle(frame, laser_pos, 5, (0, 0, 255), -1)   # Laser in Red
                cv2.line(frame, laser_pos, target_pos, (255, 255, 0), 2)
            else:
                # If target or laser is lost, we can send zeros or a specific type byte
                # 如果丢失目标或激光，可以发送 0 坐标或特定类型字节
                comm.send_data((0, 0), (0, 0), type_byte=0x00)

            
            # cv2.imshow('frame', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
except KeyboardInterrupt:
    print("Recording stopped by user.")



