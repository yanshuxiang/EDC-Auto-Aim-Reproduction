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
    2) 轮廓级候选筛选（面积）排除明显噪点与大块干扰。
    3) 候选打分（亮度 + 与预测位置距离）选最优目标。
    4) 始终在中心 ROI 区域搜索，利用物理约束抑制背景反光。
    5) 若当前帧检测不到，进入短时“预测延续”模式；超时后判定彻底丢失。

    输出兼容旧接口：
    - detect(frame) -> (x, y) 或 None

    扩展状态接口：
    - get_status() 返回当前状态（valid/confidence/lost_count/mode/position），
      便于上层控制器决定是否继续发控制量。
    """

    def __init__(
        self,
        lower=(100, 0, 170),
        upper=(170, 255, 255),
        dilate_kernel=np.ones((3, 3), np.uint8),
        min_area=2.0,
        max_area=1500.0,
        adaptive_v=True,
        v_floor=90,
        v_percentile=97,
        v_offset=55,
        roi_size=220,
        max_jump=120.0,
        max_coast_frames=6,
        use_center_roi=True,
        center_roi_ratio=0.3,
        enable_white_fallback=True,
        fallback_s_max=110,
        fallback_v_boost=8,
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

        - use_center_roi / center_roi_ratio:
          在尚未稳定跟踪时，优先只在画面中心区域搜索激光点。
          这样可以显著减少桌面边缘、背景反光的干扰。

        - enable_white_fallback / fallback_s_max / fallback_v_boost:
          低饱和高亮兜底检测。很多相机下激光点会被拍成“偏白高亮点”而不纯紫，
          该分支用于补偿这类情况：
          * fallback_s_max: 允许的最大饱和度（越小越偏“白”）
          * fallback_v_boost: 相对动态阈值再抬高一点，避免低亮度白噪声进入。

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
        self.adaptive_v = adaptive_v
        self.v_floor = int(v_floor)
        self.v_percentile = int(v_percentile)
        self.v_offset = int(v_offset)
        self.roi_size = int(roi_size)
        self.max_jump = float(max_jump)
        self.max_coast_frames = int(max_coast_frames)
        self.use_center_roi = bool(use_center_roi)
        self.center_roi_ratio = float(center_roi_ratio)
        self.enable_white_fallback = bool(enable_white_fallback)
        self.fallback_s_max = int(fallback_s_max)
        self.fallback_v_boost = int(fallback_v_boost)
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

    def _build_mask(self, hsv, allow_fallback=True):
        """
        基于 HSV 生成候选掩码。

        处理步骤：
        - 先算动态 V 下限；
        - inRange 做颜色+亮度阈值分割（主分支）；
        - 可选：低饱和高亮兜底分支（补偿“偏白激光点”）；
        - OPEN 去除孤立噪点；
        - DILATE 适度连接弱断裂区域，稳定轮廓。
        """
        v_lower = self._calc_v_lower(hsv[:, :, 2])
        lower = np.array((self.lower[0], self.lower[1], v_lower), dtype=np.uint8)
        upper = np.array(self.upper, dtype=np.uint8)
        mask_main = cv2.inRange(hsv, lower, upper)

        mask = mask_main
        if allow_fallback and self.enable_white_fallback:
            fallback_v = min(255, max(v_lower + self.fallback_v_boost, self.v_floor))
            fallback_lower = np.array((0, 0, fallback_v), dtype=np.uint8)
            fallback_upper = np.array((179, min(255, self.fallback_s_max), 255), dtype=np.uint8)
            mask_fallback = cv2.inRange(hsv, fallback_lower, fallback_upper)
            mask = cv2.bitwise_or(mask_main, mask_fallback)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.dilate_kernel, iterations=1)
        mask = cv2.dilate(mask, self.dilate_kernel, iterations=1)
        return mask

    def _center_roi_bounds(self, width, height):
        """
        计算画面中心 ROI 边界。

        ratio 取值范围会被夹到 [0.2, 1.0]：
        - 过小会非常容易漏掉目标；
        - 大于 1 没意义（等同全图）。
        """
        ratio = min(max(self.center_roi_ratio, 0.2), 1.0)
        roi_w = int(width * ratio)
        roi_h = int(height * ratio)
        x1 = max(0, (width - roi_w) // 2)
        y1 = max(0, (height - roi_h) // 2)
        x2 = min(width, x1 + roi_w)
        y2 = min(height, y1 + roi_h)
        return x1, y1, x2, y2

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
        2) 计算亮度均值与质心坐标作为后续打分输入。

        返回：
        - 候选列表（每项含 center/area/mean_v）。
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
                    "mean_v": mean_v,
                    "contour": contour,
                }
            )
        return candidates

    def _select_best(self, candidates, predicted):
        """
        对候选点进行打分，选择最佳目标。

        打分组成：
        - 亮度得分（权重 0.75）：激光通常在候选中更亮。
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
            score = 0.75 * brightness_score

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


    def detect(self, frame):
        """
        执行一次完整检测并更新内部状态。

        修改说明：
        - 始终限制在中央 ROI（center_roi）范围内搜索，不进行全图回退。
        - 理由：激光在物理意义上基本只出现在画面中央，全图搜索会增加背景干扰。

        返回：
        - (x, y): 目标坐标
        - None: 丢失状态
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        predicted = self._predict()

        # 始终在中心 ROI 检测，降低边缘背景干扰
        h, w = hsv.shape[:2]
        x1, y1, x2, y2 = self._center_roi_bounds(w, h)
        center_roi = hsv[y1:y2, x1:x2]
        center_mask = self._build_mask(center_roi, allow_fallback=True)
        candidates = self._extract_candidates(center_roi, center_mask, offset_x=x1, offset_y=y1)

        if self.isdebug:
            self.debug(center_mask)

        best, confidence = self._select_best(candidates, predicted)
        if best is not None:
            if predicted is not None:
                jump = float(np.linalg.norm(best["center"] - predicted))
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
