from PIL import Image
import cv2
import numpy as np
from dataclasses import dataclass
from itertools import combinations

class Target:
    def __init__(
        self,
        min_area=1500,
        threshold_value=120,
        kernel_size=(5, 5),
        blur_sigma=0,
        canny_low=50,
        canny_high=150,
        circle_dp=1.2,
        circle_param1=120,
        circle_param2=18,
        a4_size=(420, 297),
        debug=False
    ):
        """
        初始化目标检测器，并保存图像预处理、轮廓筛选和圆检测所需参数。

        参数大致分为三类：
        - 预处理参数：`threshold_value`、`kernel_size`、`blur_sigma`、`canny_low`、`canny_high`
        - 候选目标验证参数：`min_area`
        - 透视矫正与圆检测参数：`circle_dp`、`circle_param1`、`circle_param2`、`a4_size`

        同时会根据 `kernel_size` 预生成一次腐蚀操作使用的结构元素，避免重复创建。
        """
        self.min_area = min_area
        self.threshold_value = threshold_value
        self.kernel_size = kernel_size
        self.kernel = np.ones(kernel_size, np.uint8)
        self.blur_sigma = blur_sigma
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.circle_dp = circle_dp
        self.circle_param1 = circle_param1
        self.circle_param2 = circle_param2
        self.a4_size = a4_size
        self.isdebug = debug
        self.debug_index = 0

        print("debug:",self.isdebug)

    def preprocess(self, frame):
        """
        对输入帧执行基础预处理，生成后续轮廓检测所需的中间图像。

        处理流程：
        1. 高斯模糊，降低噪声。
        2. 转灰度图。
        3. 固定阈值二值化。
        4. Canny 边缘检测。
        5. 将二值图与边缘图进行融合。
        6. 对融合结果执行一次腐蚀，减少噪点和细碎连接。

        返回值：
        - `(binary, canny, fused, eroded)`，供调试和后续检测阶段共同使用。
        """
        blurred = cv2.GaussianBlur(frame, (5, 5), self.blur_sigma)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(gray, self.threshold_value, 255, cv2.THRESH_BINARY)[1]
        canny = cv2.Canny(gray, self.canny_low, self.canny_high)
        fused = cv2.bitwise_or(binary, canny)
        eroded=cv2.erode(fused, self.kernel, iterations=1)
        return eroded


    def extract_potential_rects(self, contours):
        # 创建 4 行空的二维数组（形状：(5, 0)）
        # rect_logs[[contours],[面积筛选],[矩形筛选],[长宽比筛选],[]]
        rect_logs = [[] for _ in range(5)]
        rect_logs[0]=contours
        potential_rects = []

        for contour in contours:
            area=cv2.contourArea(contour)
            if area < self.min_area:
                continue
            rect_logs[1].append(contour)

            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) != 4:
                continue
            rect_logs[2].append(contour)

            x, y, w, h = cv2.boundingRect(approx)
            # if w == 0 or h == 0 or w < h:
            #     continue
            # rect_logs[3].append(contour)


            potential_rects.append((contour, area,x,y,w,h))

        # potential_rects-> (contour, area,x,y,w,h)
        return rect_logs, potential_rects



    def debug(self, frame, rect_logs):
        stage_names = [
            "all_contours",
            "rejected_by_area",
            "rejected_by_vertices",
            "rejected_by_bbox",
            "final_candidates",
        ]
        debug_images = []
        for stage, name in zip(rect_logs, stage_names):
            if not stage:
                continue
            contours = stage[0] if name == "all_contours" and len(stage) == 1 else stage
            canvas = frame.copy()
            cv2.drawContours(canvas, contours, -1, (0, 0, 255), 2)
            debug_images.append(Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)))

        if not debug_images:
            return

        width, height = debug_images[0].size
        merged = Image.new("RGB", (width * len(debug_images), height), (255, 255, 255))
        for i, img in enumerate(debug_images):
            merged.paste(img, (i * width, 0))
        out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "debug", "target_merged"))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{self.debug_index:04d}.jpg")
        merged.save(out_path)
        self.debug_index += 1

    def match_rects(self,potential_rects):
        matched_pairs=[]

        for one,two in combinations(potential_rects,2):
            x1,y1,w1,h1= one[2],one[3],one[4],one[5]
            x2,y2,w2,h2= two[2],two[3],two[4],two[5]
            cx1,cx2,cy1,cy2=x1+w1/2,x2+w2/2,y1+h1/2,y2+h2/2

            x_limit = max(w1, w2) / 10
            y_limit = max(h1, h2) / 10

            x_gap_limit = max(w1, w2) / 5
            y_gap_limit = max(h1, h2) / 5

            if (
                abs(cx1 - cx2) < x_limit
                and abs(cy1 - cy2) < y_limit
                and abs(x1 - x2) <= x_gap_limit
                and abs(y1 - y2) <= y_gap_limit
            ):
                matched_pairs.append((one,two))

        return matched_pairs

    def fliter(self,frame,matched_pairs):
        final=[]
        for one,two in matched_pairs:
            if one[1]>two[1]:
                contour=one[0]
            else:
                contour=two[0]
            croped_img=self.warp_to_a4(frame,contour)
            gray=cv2.cvtColor(croped_img, cv2.COLOR_BGR2GRAY)
            binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]
            kernel = np.ones((5, 5), np.uint8)
            eroded = cv2.erode(binary, kernel, iterations=3)
            black_ratio = np.count_nonzero(eroded == 0) / eroded.size
            if black_ratio > 0.2:
                final.append((one,two))
        return final



    def detect_circle(self, frame, matched_pairs):
        pass

    def order_points(self, pts):
        """
        对四边形顶点进行固定顺序排序。

        输出顺序为：
        - 左上
        - 右上
        - 右下
        - 左下

        该顺序是透视变换所需的标准点序，便于后续将目标区域稳定地映射到 A4 平面。
        """
        ordered = np.zeros((4, 2), dtype=np.float32)
        sums = pts.sum(axis=1)
        diffs = np.diff(pts, axis=1).reshape(-1)
        ordered[0] = pts[np.argmin(sums)]
        ordered[2] = pts[np.argmax(sums)]
        ordered[1] = pts[np.argmin(diffs)]
        ordered[3] = pts[np.argmax(diffs)]
        return ordered

    def warp_to_a4(self, frame, contour):
        """
        将检测到的矩形区域透视变换到预设的 A4 尺寸平面。

        参数：
        - frame: 原图
        - contour: 轮廓

        返回值：
        - 透视矫正后的图像（A4 尺寸）
        """
        if contour is None or len(contour) < 4:
            return None

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) != 4:
            return None

        pts = approx.reshape(4, 2).astype(np.float32)
        ordered = self.order_points(pts)
        width, height = self.a4_size
        dst = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(ordered, dst)
        return cv2.warpPerspective(frame, matrix, (width, height))

    def detect(self, frame):
        preprocessed = self.preprocess(frame)
        contours, _ =cv2.findContours(preprocessed,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        rect_logs, potential_rects = self.extract_potential_rects(contours)
        matched_pairs = self.match_rects(potential_rects)

        if self.isdebug:
            self.debug(frame,rect_logs)
        if len(matched_pairs) == 0:
            print('未检测到靶子')
            return None
        if len(matched_pairs) > 1:
            matched_pairs=self.fliter(frame, matched_pairs)
        if(len(matched_pairs) > 1):
            return None

        centers = []
        for one, two in matched_pairs:
            x1, y1, w1, h1 = one[2], one[3], one[4], one[5]
            x2, y2, w2, h2 = two[2], two[3], two[4], two[5]
            centers.append(
                (
                    (int(x1 + w1 / 2), int(y1 + h1 / 2)),
                    (int(x2 + w2 / 2), int(y2 + h2 / 2)),
                )
            )
        if not centers:
            return None
        points = []
        for left, right in centers:
            points.append(left)
            points.append(right)
        avg_x = int(sum(p[0] for p in points) / len(points))
        avg_y = int(sum(p[1] for p in points) / len(points))
        return (avg_x, avg_y)


if __name__ == "__main__":
    import time
    import shutil
    import os

    debug_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "debug"))
    if os.path.isdir(debug_dir):
        for name in os.listdir(debug_dir):
            path = os.path.join(debug_dir, name)
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
    target = Target(debug=False)


    cap=cv2.VideoCapture("../media/2.mp4")
    cnt=0
    t1=time.time()
    while True:
        cnt+=1
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        # frame = cv2.resize(frame, (0, 0), fx=1 / 5, fy=1 / 5, interpolation=cv2.INTER_AREA)
        pos = target.detect(frame)
        # if res is not None:
        #     cv2.circle(frame, res, 5, (0, 0, 255), -1)
        cv2.imshow("frame", frame)
        if(cv2.waitKey(10) & 0xFF == ord('q')):
            break

    t2=time.time()
    t=t2-t1
    fps=cnt/t
    print("fps:",fps)
