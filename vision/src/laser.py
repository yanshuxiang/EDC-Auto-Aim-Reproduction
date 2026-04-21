import cv2
import numpy as np
import os
import shutil
from collections import deque


class Laser:
    """
    激光点检测与短时跟踪模块（比赛版）。

    这个类的核心目标是：在复杂光照和反光干扰场景下，尽可能稳定地给出激光点坐标，
    并在短时间丢失目标时避免输出突然中断或剧烈跳变。

    检测+跟踪主流程概览：
    1) 颜色与亮度阈值分割（HSV）得到候选掩码。
    2) 轮廓级候选筛选（面积、圆度、实心度）排除噪点与非激光斑。
    3) 候选打分（亮度 + 形状 + 与预测位置距离）选最优目标。
    4) 若连续帧稳定，优先在 ROI（局部区域）搜索，提高速度和抗干扰能力。
    5) 若当前帧检测不到，进入短时“预测延续”模式；超时后判定彻底丢失。

    输出兼容旧接口：
    - detect(frame) -> (x, y) 或 None

    扩展状态接口：
    - get_status() 返回当前状态（valid/confidence/lost_count/mode/position），
      便于上层控制器决定是否继续发控制量。
    """

    def __init__(
        self,
        lower=(120, 30, 220),
        upper=(160, 255, 255),
        dilate_kernel=np.ones((3, 3), np.uint8),
        min_area=5.0,
        max_area=500.0,
        min_circularity=0.35,
        min_solidity=0.5,
        adaptive_v=True,
        v_floor=120,
        v_percentile=98,
        v_offset=35,
        roi_size=160,
        max_jump=80.0,
        max_coast_frames=4,
        ema_alpha=0.65,
        isdebug=False,
    ):
        """
        初始化检测器与跟踪器参数。

        参数详细说明：
        - lower / upper:
          HSV 阈值范围（H, S, V）。H 是色相，S 是饱和度，V 是亮度。
          这里主要用于筛选“紫色/偏紫高亮”激光点。

        - dilate_kernel:
          形态学核。先做 OPEN 再做 DILATE，用于去小噪点、连接断裂区域。

        - min_area / max_area:
          轮廓面积过滤区间（像素面积）。
          太小一般是噪声，太大通常是灯带/反光斑/大面积干扰。

        - min_circularity:
          最小圆度阈值。圆度公式为 4πA/P²，越接近 1 越像圆。
          激光点通常比较接近圆斑，圆度过低常见于条纹反光。

        - min_solidity:
          最小实心度阈值。实心度 = area / convex_hull_area。
          可过滤“空心/破碎/锯齿”轮廓。

        - adaptive_v:
          是否启用亮度阈值自适应（仅自适应 V 下限）。
          开启后在不同环境亮度下更稳，减少“太亮场景下全都过阈值”的误检。

        - v_floor:
          自适应 V 下限的最小值，防止阈值过低导致噪声泛滥。

        - v_percentile:
          场景亮度峰值估计采用的分位数（例如 98 表示取 V 通道的 98 分位）。

        - v_offset:
          动态阈值 = 峰值 - 偏移量。偏移越大，阈值越严格。

        - roi_size:
          局部搜索窗口边长（像素）。跟踪稳定时优先在该窗口内找目标。

        - max_jump:
          单帧允许的最大跳变（像素）。超过则认为可疑，不立即接收该测量。

        - max_coast_frames:
          最大“无测量预测延续”帧数。超过后进入 lost 状态并返回 None。

        - ema_alpha:
          位置更新的指数平滑权重。越大越跟随新测量，越小越平滑。

        - isdebug:
          是否保存调试掩码图（会写入 debug 目录，可能影响性能）。
        """
        self.lower = tuple(lower)
        self.upper = tuple(upper)
        self.dilate_kernel = dilate_kernel
        self.min_area = float(min_area)
        self.max_area = float(max_area)
        self.min_circularity = float(min_circularity)
        self.min_solidity = float(min_solidity)
        self.adaptive_v = adaptive_v
        self.v_floor = int(v_floor)
        self.v_percentile = int(v_percentile)
        self.v_offset = int(v_offset)
        self.roi_size = int(roi_size)
        self.max_jump = float(max_jump)
        self.max_coast_frames = int(max_coast_frames)
        self.ema_alpha = float(ema_alpha)
        self.isdebug = isdebug
        self.debug_index = 0

        # -------------------- 跟踪状态 --------------------
        # last_pos: 上一时刻估计位置（float，便于平滑）
        self.last_pos = None
        # last_vel: 上一时刻估计速度（像素/帧）
        self.last_vel = np.zeros(2, dtype=np.float32)
        # lost_count: 连续“没有可靠测量”的帧数
        self.lost_count = 0
        # v_history: 动态亮度阈值估计历史（用于抗闪烁）
        self.v_history = deque(maxlen=20)
        # last_result: 对外可读状态快照（给上层控制器使用）
        self.last_result = {
            "valid": False,
            "confidence": 0.0,
            "lost_count": 0,
            "mode": "init",
            "position": None,
        }

    def debug(self, frame):
        """
        保存调试图（通常是二值掩码）。

        说明：
        - 该函数每调用一次会保存一张图片。
        - 频繁开启会增加 IO 开销，比赛实战建议默认关闭。
        """
        self.debug_index += 1
        out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "debug", "laser"))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{self.debug_index:04d}.jpg")
        cv2.imwrite(out_path, frame)

    def get_status(self):
        """
        返回最近一次 detect() 的状态副本。

        字段说明：
        - valid: 当前输出是否可用于控制。
        - confidence: 0~1，检测置信分值（启发式，不是概率学严格定义）。
        - lost_count: 当前已连续丢失多少帧。
        - mode: measured / predicted / lost / init
          measured: 本帧有真实测量
          predicted: 本帧无测量，使用短时预测值
          lost: 超过最大续跟帧，彻底判丢失
          init: 初始化未运行
        - position: 当前输出坐标（或 None）
        """
        return dict(self.last_result)

    def _calc_v_lower(self, v_channel):
        """
        计算 V（亮度）通道下限阈值。

        两种模式：
        1) 非自适应：直接使用 lower[2]。
        2) 自适应：
           - 取当前帧 V 通道高分位数，估计场景亮区峰值；
           - 与历史中位数融合，减小闪烁影响；
           - 减去 offset 后得到动态下限；
           - 再夹到 [v_floor, upper_v-1] 范围，避免阈值越界。
        """
        if not self.adaptive_v:
            return int(self.lower[2])

        scene_peak = np.percentile(v_channel, self.v_percentile)
        self.v_history.append(float(scene_peak))
        smoothed_peak = np.median(self.v_history)
        dynamic_v = int(smoothed_peak - self.v_offset)
        dynamic_v = max(self.v_floor, dynamic_v)
        dynamic_v = min(dynamic_v, int(self.upper[2]) - 1)
        return max(dynamic_v, 0)

    def _build_mask(self, hsv):
        """
        基于 HSV 生成候选掩码。

        处理步骤：
        - 先算动态 V 下限；
        - inRange 做颜色+亮度阈值分割；
        - OPEN 去除孤立噪点；
        - DILATE 适度连接弱断裂区域，稳定轮廓。
        """
        v_lower = self._calc_v_lower(hsv[:, :, 2])
        lower = np.array((self.lower[0], self.lower[1], v_lower), dtype=np.uint8)
        upper = np.array(self.upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.dilate_kernel, iterations=1)
        mask = cv2.dilate(mask, self.dilate_kernel, iterations=1)
        return mask

    def _extract_candidates(self, hsv, mask, offset_x=0, offset_y=0):
        """
        从掩码中提取并筛选候选点。

        参数：
        - hsv: 对应区域的 HSV 图像。
        - mask: 对应区域的二值掩码。
        - offset_x / offset_y:
          当 mask 来自 ROI（局部裁剪）时，需要把局部坐标映射回全图坐标。

        候选筛选规则：
        1) 面积区间过滤。
        2) 圆度过滤。
        3) 实心度过滤。
        4) 计算亮度均值与质心坐标作为后续打分输入。

        返回：
        - 候选列表（每项含 center/area/circularity/solidity/mean_v）。
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        candidates = []
        v_channel = hsv[:, :, 2]
        contour_mask = np.zeros(mask.shape, dtype=np.uint8)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue

            peri = cv2.arcLength(contour, True)
            if peri <= 1e-6:
                continue

            circularity = float((4.0 * np.pi * area) / (peri * peri))
            if circularity < self.min_circularity:
                continue

            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area <= 1e-6:
                continue
            solidity = float(area / hull_area)
            if solidity < self.min_solidity:
                continue

            contour_mask.fill(0)
            cv2.drawContours(contour_mask, [contour], -1, 255, -1)
            mean_v = float(cv2.mean(v_channel, mask=contour_mask)[0])

            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = float(M["m10"] / M["m00"]) + float(offset_x)
            cy = float(M["m01"] / M["m00"]) + float(offset_y)

            candidates.append(
                {
                    "center": np.array([cx, cy], dtype=np.float32),
                    "area": float(area),
                    "circularity": circularity,
                    "solidity": solidity,
                    "mean_v": mean_v,
                    "contour": contour,
                }
            )
        return candidates

    def _select_best(self, candidates, predicted):
        """
        对候选点进行打分，选择最佳目标。

        打分组成：
        - 亮度得分（权重 0.65）：激光通常在候选中更亮。
        - 形状得分（权重 0.25）：圆度与实心度平均。
        - 距离得分（权重 0.25，可选）：越接近预测位置越可信。

        说明：
        - 这里是启发式评分，不是学习模型概率。
        - 最终 confidence 取分数夹紧到 [0,1]。
        """
        if not candidates:
            return None, 0.0

        best_candidate = None
        best_score = -1.0
        for item in candidates:
            brightness_score = min(max(item["mean_v"] / 255.0, 0.0), 1.0)
            shape_score = 0.5 * min(item["circularity"], 1.0) + 0.5 * min(item["solidity"], 1.0)
            score = 0.65 * brightness_score + 0.25 * shape_score

            if predicted is not None:
                dist = float(np.linalg.norm(item["center"] - predicted))
                distance_score = max(0.0, 1.0 - dist / max(self.max_jump, 1.0))
                score += 0.25 * distance_score

            if score > best_score:
                best_score = score
                best_candidate = item

        confidence = min(max(best_score, 0.0), 1.0)
        return best_candidate, confidence

    def _predict(self):
        """
        线性运动预测：位置 = 上一位置 + 上一速度。

        这是一个极简常速度模型，优点是计算开销非常低，适合高帧率实时环节。
        """
        if self.last_pos is None:
            return None
        return self.last_pos + self.last_vel

    def _update_state_with_measurement(self, measured, confidence):
        """
        用“真实测量值”更新状态。

        关键点：
        - 速度估计：新速度与旧速度做线性融合，减小抖动。
        - 位置估计：用 EMA 在“新测量”和“预测位置”间折中，平滑输出。
        - 丢失计数清零，模式置为 measured。
        """
        measured = measured.astype(np.float32)

        if self.last_pos is None:
            # 首次测量：直接建轨迹
            self.last_pos = measured
            self.last_vel = np.zeros(2, dtype=np.float32)
        else:
            # raw_vel: 当前测量相对上一估计的位置增量（像素/帧）
            raw_vel = measured - self.last_pos
            # 速度平滑，避免噪声让预测速度剧烈抖动
            self.last_vel = 0.55 * self.last_vel + 0.45 * raw_vel
            predicted = self._predict()
            if predicted is None:
                self.last_pos = measured
            else:
                # 位置平滑：预测值负责连续性，测量值负责纠偏
                self.last_pos = self.ema_alpha * measured + (1.0 - self.ema_alpha) * predicted

        self.lost_count = 0
        output = tuple(int(round(v)) for v in self.last_pos.tolist())
        self.last_result = {
            "valid": True,
            "confidence": float(confidence),
            "lost_count": self.lost_count,
            "mode": "measured",
            "position": output,
        }
        return output

    def _update_state_without_measurement(self):
        """
        在“本帧无可靠测量”时更新状态。

        策略分两段：
        1) 若还在 max_coast_frames 内：
           - 使用 last_pos + last_vel 做短时预测续跟；
           - confidence 随丢帧数衰减；
           - 模式为 predicted，输出仍给坐标（防止控制环突变）。
        2) 超过 max_coast_frames：
           - 判定彻底丢失；
           - 清空轨迹状态；
           - 模式为 lost，输出 None。
        """
        if self.last_pos is not None and self.lost_count < self.max_coast_frames:
            self.lost_count += 1
            self.last_pos = self.last_pos + self.last_vel
            # 随着连续丢失增加，置信度线性下降
            decay = 1.0 - self.lost_count / (self.max_coast_frames + 1.0)
            confidence = max(0.0, 0.35 * decay)
            output = tuple(int(round(v)) for v in self.last_pos.tolist())
            self.last_result = {
                "valid": True,
                "confidence": float(confidence),
                "lost_count": self.lost_count,
                "mode": "predicted",
                "position": output,
            }
            return output

        self.lost_count = min(self.lost_count + 1, self.max_coast_frames + 5)
        self.last_pos = None
        self.last_vel = np.zeros(2, dtype=np.float32)
        self.last_result = {
            "valid": False,
            "confidence": 0.0,
            "lost_count": self.lost_count,
            "mode": "lost",
            "position": None,
        }
        return None

    def _try_roi_detection(self, hsv, predicted):
        """
        在预测位置周围做局部 ROI 检测。

        作用：
        - 降低计算量；
        - 限制空间范围，减少“远处强反光”对当前目标的抢占。

        返回：
        - candidates: ROI 内候选（已映射为全图坐标）
        - mask: ROI 对应掩码（用于 debug）
        """
        if predicted is None:
            return [], None

        height, width = hsv.shape[:2]
        half = self.roi_size // 2
        px = int(round(predicted[0]))
        py = int(round(predicted[1]))
        x1 = max(0, px - half)
        y1 = max(0, py - half)
        x2 = min(width, px + half)
        y2 = min(height, py + half)

        if x2 <= x1 or y2 <= y1:
            return [], None

        roi = hsv[y1:y2, x1:x2]
        mask = self._build_mask(roi)
        return self._extract_candidates(roi, mask, offset_x=x1, offset_y=y1), mask

    def detect(self, frame):
        """
        执行一次完整检测并更新内部状态。

        主过程：
        1) 计算预测位置（若轨迹存在）。
        2) 如果轨迹有效，先尝试 ROI 局部检测；失败则回退全图检测。
        3) 对候选打分选最优。
        4) 若有候选，先做“突变门限”检查：
           - 若相对预测跳变过大，判为异常测量，走无测量分支；
           - 否则接收测量并更新状态。
        5) 若无候选，走无测量分支（短时预测续跟或最终 lost）。

        返回：
        - (x, y): measured 或 predicted 模式下的输出坐标
        - None: lost 模式下输出
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        predicted = self._predict()

        candidates = []
        debug_mask = None

        if self.last_pos is not None and self.lost_count <= self.max_coast_frames:
            candidates, roi_mask = self._try_roi_detection(hsv, predicted)
            if roi_mask is not None:
                debug_mask = roi_mask

        if not candidates:
            full_mask = self._build_mask(hsv)
            candidates = self._extract_candidates(hsv, full_mask)
            debug_mask = full_mask

        if self.isdebug and debug_mask is not None:
            self.debug(debug_mask)

        best, confidence = self._select_best(candidates, predicted)
        if best is not None:
            if predicted is not None:
                jump = float(np.linalg.norm(best["center"] - predicted))
                # 跳变过大通常是误检（例如突然切到别的反光点），先拒绝该测量
                if jump > self.max_jump:
                    return self._update_state_without_measurement()
            return self._update_state_with_measurement(best["center"], confidence)

        return self._update_state_without_measurement()


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
