import cv2
import numpy as np
import time
import os

class Target:
    def __init__(
        self,
        min_area=1500,
        threshold_value=120,
        kernel_size=(3, 3),
        blur_sigma=0,
        canny_low=50,
        canny_high=150,
        a4_long_mm=297.0,
        a4_short_mm=210.0,
        frame_width_mm=18.0,
        ring_black_threshold=90,
        ring_black_min_ratio=0.2,
        min_ring_ratio=0.015,
        inner_white_threshold=145,
        white_weight=0.5,
        ring_weight=0.5,
        debug=False,
        debug_save_images=False,
        debug_dir=None,
        debug_print_every=1,
    ):
        """
        初始化目标检测器，并保存图像预处理、轮廓筛选和圆检测所需参数。

        参数大致分为两类：
        - 预处理参数：`threshold_value`、`kernel_size`、`blur_sigma`、`canny_low`、`canny_high`
        - 候选目标验证参数：`min_area`
        - 靶纸几何参数：`a4_long_mm`、`a4_short_mm`、`frame_width_mm`
        - 环带验证参数：`ring_black_threshold`、`ring_black_min_ratio`

        同时会根据 `kernel_size` 预生成一次腐蚀操作使用的结构元素，避免重复创建。
        """
        self.min_area = min_area
        self.threshold_value = threshold_value
        self.kernel_size = kernel_size
        self.kernel = np.ones(kernel_size, np.uint8)
        self.blur_sigma = blur_sigma
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.a4_long_mm = float(a4_long_mm)
        self.a4_short_mm = float(a4_short_mm)
        self.frame_width_mm = float(frame_width_mm)
        self.inner_long_mm = self.a4_long_mm - 2.0 * self.frame_width_mm
        self.inner_short_mm = self.a4_short_mm - 2.0 * self.frame_width_mm
        if self.inner_long_mm <= 1e-6 or self.inner_short_mm <= 1e-6:
            raise ValueError("frame_width_mm is too large for A4 dimensions")
        self.ring_black_threshold = int(ring_black_threshold)
        self.ring_black_min_ratio = float(ring_black_min_ratio)
        self.min_ring_ratio = float(min_ring_ratio)
        self.inner_white_threshold = int(inner_white_threshold)
        self.white_weight = float(white_weight)
        self.ring_weight = float(ring_weight)
        total_weight = self.white_weight + self.ring_weight
        if total_weight <= 1e-9:
            self.white_weight = 0.5
            self.ring_weight = 0.5
        else:
            # 归一化，确保两项按相对权重稳定融合
            self.white_weight /= total_weight
            self.ring_weight /= total_weight
        self.debug = bool(debug)
        self.debug_save_images = bool(debug_save_images)
        self.debug_print_every = max(int(debug_print_every), 1)
        default_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "debug", "target_filter")
        )
        self.debug_dir = debug_dir if debug_dir is not None else default_dir
        self.debug_frame_index = 0
        self.last_debug_info = {}
        if self.debug_save_images:
            os.makedirs(self.debug_dir, exist_ok=True)

    def _order_points(self, pts):
        """
        将四点排序为：左上、右上、右下、左下。
        """
        pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
        ordered = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).reshape(-1)
        ordered[0] = pts[np.argmin(s)]
        ordered[2] = pts[np.argmax(s)]
        ordered[1] = pts[np.argmin(d)]
        ordered[3] = pts[np.argmax(d)]
        return ordered

    def _project_outer_quad_from_inner(self, inner_pts):
        """
        通过单应矩阵把“内框四点”映射到内平面，再按物理尺寸扩展到外框，
        最后逆映射回原图，得到透视一致的外框四点。
        """
        inner_img = self._order_points(inner_pts)

        inner_w = float(self.inner_long_mm)
        inner_h = float(self.inner_short_mm)
        outer_w = float(self.a4_long_mm)
        outer_h = float(self.a4_short_mm)
        border = float(self.frame_width_mm)

        # 内框在“毫米平面”中的坐标
        inner_plane = np.array(
            [[0.0, 0.0], [inner_w, 0.0], [inner_w, inner_h], [0.0, inner_h]],
            dtype=np.float32,
        )

        # 外框在“毫米平面”中的坐标（向四周各扩 border）
        outer_plane = np.array(
            [
                [-border, -border],
                [inner_w + border, -border],
                [inner_w + border, inner_h + border],
                [-border, inner_h + border],
            ],
            dtype=np.float32,
        )

        H = cv2.getPerspectiveTransform(inner_plane, inner_img)
        outer_img = cv2.perspectiveTransform(outer_plane.reshape(-1, 1, 2), H).reshape(4, 2)
        return outer_img

    def _line_intersection(self, p1, p2, p3, p4):
        """
        计算两条直线 (p1,p2) 与 (p3,p4) 的交点。
        若两线近似平行，返回 None。
        """
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
        x3, y3 = float(p3[0]), float(p3[1])
        x4, y4 = float(p4[0]), float(p4[1])

        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < 1e-6:
            return None

        det1 = x1 * y2 - y1 * x2
        det2 = x3 * y4 - y3 * x4
        px = (det1 * (x3 - x4) - (x1 - x2) * det2) / den
        py = (det1 * (y3 - y4) - (y1 - y2) * det2) / den
        return np.array([px, py], dtype=np.float32)

    def _center_from_quad_diagonal(self, quad):
        """
        使用四边形两条对角线交点作为中心点（抗透视）。
        """
        pts = self._order_points(np.asarray(quad, dtype=np.float32).reshape(4, 2))
        center = self._line_intersection(pts[0], pts[2], pts[1], pts[3])
        if center is None:
            # 退化情况兜底：极少数近平行时用点均值
            center = np.mean(pts, axis=0)
        return center

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
        - 预处理后的二值结果，用于后续轮廓检测阶段。
        """
        blurred = cv2.GaussianBlur(frame, (5, 5), self.blur_sigma)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(gray, self.threshold_value, 255, cv2.THRESH_BINARY)[1]
        canny = cv2.Canny(gray, self.canny_low, self.canny_high)
        fused = cv2.bitwise_or(binary, canny)
        eroded=cv2.erode(fused, self.kernel, iterations=1)
        return eroded


    def extract_potential_rects(self, contours):
        potential_rects = []

        for contour in contours:
            area=cv2.contourArea(contour)
            if area < self.min_area:
                continue

            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            # 优先使用轮廓近似得到四边形；失败时退化为最小外接旋转矩形
            # 避免因边缘轻微粘连导致“顶点数 != 4”被直接丢弃
            if len(approx) == 4:
                quad = approx
            else:
                continue

            x, y, w, h = cv2.boundingRect(quad)
            # if w == 0 or h == 0 or w < h:
            #     continue
            # rect_logs[3].append(contour)

            source = "approx4" if len(approx) == 4 else "minrect_fallback"
            potential_rects.append((contour, area, x, y, w, h, quad, source))

        return potential_rects

    def score_candidate(self, frame, candidate):
        """
        给单个候选矩形打分，分数越高置信度越高。

        评分策略：
        - 只基于“内框外扩后的环带区域”计算黑色占比。
        - 不再使用整块透视图黑占比（因为该逻辑隐含外框检测成功前提）。
        - 外扩尺度由实物尺寸决定：A4(297x210mm) + 黑框宽18mm。
        """
        quad = candidate[6]
        area = candidate[1]
        source = candidate[7]

        inner_pts = np.array(quad, dtype=np.float32).reshape(4, 2)
        outer_pts = self._project_outer_quad_from_inner(inner_pts)

        h, w = frame.shape[:2]
        outer_pts[:, 0] = np.clip(outer_pts[:, 0], 0, w - 1)
        outer_pts[:, 1] = np.clip(outer_pts[:, 1], 0, h - 1)

        inner_poly = np.round(inner_pts).astype(np.int32)
        outer_poly = np.round(outer_pts).astype(np.int32)

        inner_mask = np.zeros((h, w), dtype=np.uint8)
        outer_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(inner_mask, [inner_poly], 255)
        cv2.fillPoly(outer_mask, [outer_poly], 255)
        ring_mask = cv2.bitwise_and(outer_mask, cv2.bitwise_not(inner_mask))
        inner_pixels = int(np.count_nonzero(inner_mask))
        if inner_pixels <= 0:
            return None, {
                "reason": "inner_pixels",
                "ring_pixels": 0,
                "ring_ratio": 0.0,
                "black_ratio": 0.0,
                "white_ratio": 0.0,
                "area": float(area),
                "source": source,
                "inner_poly": inner_poly,
                "outer_poly": outer_poly,
            }

        ring_pixels = int(np.count_nonzero(ring_mask))
        ring_ratio = ring_pixels / inner_pixels
        if ring_ratio < self.min_ring_ratio:
            return None, {
                "reason": "ring_ratio",
                "ring_pixels": ring_pixels,
                "ring_ratio": float(ring_ratio),
                "black_ratio": 0.0,
                "white_ratio": 0.0,
                "area": float(area),
                "source": source,
                "inner_poly": inner_poly,
                "outer_poly": outer_poly,
            }

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 重点新增：在“初始内框”上计算白色像素占比，并作为高权重条件。
        white_mask = np.zeros((h, w), dtype=np.uint8)
        white_mask[gray >= self.inner_white_threshold] = 255
        white_in_inner = cv2.bitwise_and(white_mask, inner_mask)
        white_ratio = np.count_nonzero(white_in_inner) / inner_pixels
        black_mask = np.zeros((h, w), dtype=np.uint8)
        black_mask[gray <= self.ring_black_threshold] = 255
        black_in_ring = cv2.bitwise_and(black_mask, ring_mask)
        black_ratio = np.count_nonzero(black_in_ring) / ring_pixels

        if black_ratio <= self.ring_black_min_ratio:
            return None, {
                "reason": "black_ratio",
                "ring_pixels": ring_pixels,
                "ring_ratio": float(ring_ratio),
                "black_ratio": float(black_ratio),
                "white_ratio": float(white_ratio),
                "area": float(area),
                "source": source,
                "inner_poly": inner_poly,
                "outer_poly": outer_poly,
            }

        # 白色占比权重更大：优先选择内框更白的候选，再结合黑框环带约束。
        quality = self.white_weight * white_ratio + self.ring_weight * black_ratio
        score = quality
        return score, {
            "reason": "passed",
            "ring_pixels": ring_pixels,
            "ring_ratio": float(ring_ratio),
            "black_ratio": float(black_ratio),
            "white_ratio": float(white_ratio),
            "area": float(area),
            "score": float(score),
            "source": source,
            "inner_poly": inner_poly,
            "outer_poly": outer_poly,
        }

    def get_last_debug_info(self):
        return dict(self.last_debug_info)

    def _render_debug_frame(self, frame, candidate_debugs, best_idx):
        canvas = frame.copy()
        for i, item in enumerate(candidate_debugs):
            inner = item["inner_poly"]
            outer = item["outer_poly"]
            reason = item["reason"]
            color = (0, 255, 0) if reason == "passed" else (0, 0, 255)
            if reason == "black_ratio":
                color = (0, 165, 255)
            if i == best_idx:
                color = (255, 255, 0)
            cv2.polylines(canvas, [outer], True, color, 2)
            cv2.polylines(canvas, [inner], True, color, 2)
            x, y, w, h = cv2.boundingRect(inner)
            label = f"#{i} {reason}"
            if "black_ratio" in item:
                label += f" br={item['black_ratio']:.2f}"
            if "white_ratio" in item:
                label += f" wr={item['white_ratio']:.2f}"
            if "ring_ratio" in item:
                label += f" rp={item['ring_ratio']:.2f}"
            cv2.putText(
                canvas,
                label,
                (x, max(15, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

        out_path = os.path.join(self.debug_dir, f"{self.debug_frame_index:06d}.jpg")
        cv2.imwrite(out_path, canvas)

    def detect(self, frame):
        preprocessed = self.preprocess(frame)
        contours, _ =cv2.findContours(preprocessed,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        potential_rects = self.extract_potential_rects(contours)
        self.debug_frame_index += 1

        debug_info = {
            "frame_index": self.debug_frame_index,
            "raw_contours": len(contours),
            "candidates_after_area": len(potential_rects),
            "source_counts": {"approx4": 0, "minrect_fallback": 0},
            "reject_counts": {"inner_pixels": 0, "ring_ratio": 0, "black_ratio": 0},
            "passed_count": 0,
            "selected_candidate": None,
        }
        for c in potential_rects:
            debug_info["source_counts"][c[7]] += 1

        if len(potential_rects) == 0:
            self.last_debug_info = debug_info
            if self.debug and self.debug_frame_index % self.debug_print_every == 0:
                print(f"[TargetDebug] frame={self.debug_frame_index} raw={len(contours)} cand=0 -> no_target")
            return None

        # 多候选不再直接失败，而是按分数选最优
        scored = []
        candidate_debugs = []
        for idx, candidate in enumerate(potential_rects):
            score, detail = self.score_candidate(frame, candidate)
            detail["candidate_index"] = idx
            candidate_debugs.append(detail)
            if score is not None:
                scored.append((score, candidate, idx, detail))
                debug_info["passed_count"] += 1
            else:
                if detail["reason"] not in debug_info["reject_counts"]:
                    debug_info["reject_counts"][detail["reason"]] = 0
                debug_info["reject_counts"][detail["reason"]] += 1

        if not scored:
            self.last_debug_info = debug_info
            if self.debug and self.debug_frame_index % self.debug_print_every == 0:
                print(
                    f"[TargetDebug] frame={self.debug_frame_index} raw={debug_info['raw_contours']} "
                    f"cand={debug_info['candidates_after_area']} reject={debug_info['reject_counts']} -> no_target"
                )
            if self.debug and self.debug_save_images:
                self._render_debug_frame(frame, candidate_debugs, best_idx=-1)
            return None

        best_score, best_candidate, best_idx, best_detail = max(scored, key=lambda item: item[0])
        debug_info["selected_candidate"] = {
            "candidate_index": int(best_idx),
            "score": float(best_score),
            "white_ratio": float(best_detail["white_ratio"]),
            "black_ratio": float(best_detail["black_ratio"]),
            "ring_pixels": int(best_detail["ring_pixels"]),
            "ring_ratio": float(best_detail["ring_ratio"]),
            "source": best_detail["source"],
        }
        self.last_debug_info = debug_info
        if self.debug and self.debug_frame_index % self.debug_print_every == 0:
            print(
                f"[TargetDebug] frame={self.debug_frame_index} raw={debug_info['raw_contours']} "
                f"cand={debug_info['candidates_after_area']} pass={debug_info['passed_count']} "
                f"reject={debug_info['reject_counts']} select={debug_info['selected_candidate']}"
            )
        if self.debug and self.debug_save_images:
            self._render_debug_frame(frame, candidate_debugs, best_idx=best_idx)

        center = self._center_from_quad_diagonal(best_candidate[6])
        return (int(round(center[0])), int(round(center[1])))


if __name__ == "__main__":
    target = Target()


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
