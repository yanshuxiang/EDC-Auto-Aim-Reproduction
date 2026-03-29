import cv2
import numpy as np

class Laser:
    def __init__(self,
        lower=(120, 30, 220),
        upper = (160, 255, 255),
        dilate_kernel=(5,5)
        ):
        self.lower=lower
        self.upper=upper
        self.dilate_kernel=dilate_kernel

    def detect(self,frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        dilated = cv2.dilate(mask, self.dilate_kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 1 or len(contours) ==0 :
            print("contour数量出错")
            return None
        M = cv2.moments(contours[0])
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        return(cx,cy)


if __name__ == '__main__':
    cap = cv2.VideoCapture("../media/1.mp4")
    laser = Laser()
    while True:
        ret,frame=cap.read()
        if not(ret):
            break
        frame=cv2.resize(frame,(0,0),fx=1/2,fy=1/2)
        pos=laser.detect(frame)
        cv2.circle(frame,pos,5,(0,0,255),2)
        cv2.imshow("frame",frame)
        if(cv2.waitKey(10) & 0xFF == ord('q')):
            break
    cap.release()
    cv2.destroyAllWindows()
