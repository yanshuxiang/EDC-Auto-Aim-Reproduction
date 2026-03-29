import cv2
import numpy as np

cap=cv2.VideoCapture("../media/1.mp4")

while True:
    ret,frame=cap.read()
    if not(ret):
        break
    frame=cv2.resize(frame,(0,0),fx=1/3,fy=1/3,interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = (120, 30, 220)
    upper = (160, 255, 255)
    mask = cv2.inRange(hsv, lower, upper)
    dilated = cv2.dilate(mask, (5,5), iterations=2)
    # cv2.imshow("frame",frame)
    cv2.imshow("purple_mask", dilated)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
