import cv2
import numpy as np
from vision.src.laser import Laser
from vision.src.target import Target
from vision.src.capture import FrameCapture
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

t=time.time()
fps=0

with FrameCapture(23) as capture:
    while True:
        fps+=1
        t1=time.time()
        if(t1-t>1):
            print(fps)
            t=time.time()
            fps=0
        
        ret, frame = capture.read()
        if not ret:
            break

        if debug:
            init_dir()

        # target_pos=target.detect(frame)
        # laser_pos=laser.detect(frame)
        # cv2.circle(frame, target_pos, 5, (0,0,255), -1)
        # if(target_pos is not None):
        #     print(target_pos)
        #     cv2.circle(frame,target_pos,3,(255,0,0),-1)
        # cv2.imwrite("test.jpg",frame)
        # cv2.imshow('frame',frame)
        # if(cv2.waitKey(10) & 0xFF == ord('q')):
            # break
        # if laser_pos is not None:
        #     print("检测到激光")
        # else:
        #     print('未检测到激光')
        # pass




