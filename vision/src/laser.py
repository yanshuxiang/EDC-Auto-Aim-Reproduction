import cv2
import numpy as np
import os
import shutil


class Laser:
    def __init__(
        self,
        lower=(120, 30, 220),
        upper=(160, 255, 255),
        dilate_kernel=np.ones((5, 5), np.uint8),
        isdebug=False
    ):
        self.lower = lower
        self.upper = upper
        self.dilate_kernel = dilate_kernel
        self.isdebug = isdebug
        self.debug_index = 0

    def debug(self, frame):
        self.debug_index += 1
        out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "debug", "laser"))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{self.debug_index:04d}.jpg")
        cv2.imwrite(out_path, frame)

    def detect(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        dilated = cv2.dilate(mask, self.dilate_kernel, iterations=2)
        if self.isdebug:
            self.debug(dilated)

        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) != 1:
            print("contour数量错误")
            return None

        M = cv2.moments(contours[0])
        if M["m00"] == 0:
            return None

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)


if __name__ == '__main__':
    cap = cv2.VideoCapture("../media/1.mp4")
    laser = Laser(isdebug=True)
    debug_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "debug"))
    if os.path.isdir(debug_dir):
        for name in os.listdir(debug_dir):
            path = os.path.join(debug_dir, name)
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (0, 0), fx=1 / 2, fy=1 / 2)
        pos = laser.detect(frame)
        if pos is not None:
            cv2.circle(frame, pos, 5, (0, 0, 255), 2)
        cv2.imshow("frame", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
