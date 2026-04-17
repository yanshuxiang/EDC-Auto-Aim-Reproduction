import cv2
import numpy as np
from vision.src.laser import Laser
from vision.src.target import Target
from vision.src.capture import FrameCapture
from vision.src.saver import ResultSaver
from vision.src.pid import PID
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

# Initialize PID controllers for X (Yaw) and Y (Pitch)
# 初始化 X (水平/Yaw) 和 Y (垂直/Pitch) 的 PID 控制器
pid_x = PID(kp=0.5, ki=0.01, kd=0.05, output_limit=1000, integral_limit=500)
pid_y = PID(kp=0.5, ki=0.01, kd=0.05, output_limit=1000, integral_limit=500)

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
            
            # Control logic: only act if both target and laser are detected
            # 控制逻辑：仅在同时检测到靶子和激光时执行
            if target_pos is not None and laser_pos is not None:
                # Calculate error (Target - Laser)
                # 计算偏差 (目标 - 激光)
                error_x = target_pos[0] - laser_pos[0]
                error_y = target_pos[1] - laser_pos[1]
                
                # Update PID controllers
                # 更新 PID 控制器
                out_x = pid_x.update(error_x)
                out_y = pid_y.update(error_y)
                
                # Send control commands via serial
                # 通过串口发送控制指令
                comm.send_control(out_x, out_y)
                
                if debug:
                    print(f"Aligning: ErroX={error_x}, ErroY={error_y} -> OutX={out_x:.2f}, OutY={out_y:.2f}")

                # Visualization (Optional)
                cv2.circle(frame, target_pos, 5, (0, 255, 0), -1)  # Target in Green
                cv2.circle(frame, laser_pos, 5, (0, 0, 255), -1)   # Laser in Red
                cv2.line(frame, laser_pos, target_pos, (255, 255, 0), 2)
            else:
                # If target or laser is lost, reset PID state to prevent windup
                # 如果丢失目标，重置 PID 状态防止积分饱和
                pid_x.reset()
                pid_y.reset()
                # Optionally send stop command if needed
                # comm.send_control(0, 0)
            
            # cv2.imshow('frame', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
except KeyboardInterrupt:
    print("Recording stopped by user.")



