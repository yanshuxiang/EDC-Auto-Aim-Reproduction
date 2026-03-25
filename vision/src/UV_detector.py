import cv2
import numpy as np

cap=cv2.VideoCapture("../media/1.mp4")

import cv2
import numpy as np

def detect_laser_fast(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 1️⃣ 模糊亮度（防噪）
    v_blur = cv2.GaussianBlur(v, (3, 3), 0)

    # 2️⃣ 找最亮点
    _, max_val, _, max_loc = cv2.minMaxLoc(v_blur)

    # 3️⃣ 亮度过滤（第一层）
    if max_val < 100:
        return None

    x, y = max_loc

    # 4️⃣ 取该点 HSV
    h_val = h[y, x]
    s_val = s[y, x]
    v_val = v[y, x]

    # 5️⃣ 紫色（宽范围！！）
    # OpenCV H: 0~179
    # 紫色大致分布在 110~170（蓝紫到紫红）
    if not (110 <= h_val <= 170 and s_val >= 50):
        return None

    return (x, y)

while True:
    ret,frame=cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    loc=detect_laser_fast(frame)
    cv2.circle(frame,loc,5,(0,0,255),-1)
    cv2.imshow("frame",frame)
    if(cv2.waitKey(30) & 0xFF == ord('q')):
        break

cv2.destroyAllWindows()
cap.release()
