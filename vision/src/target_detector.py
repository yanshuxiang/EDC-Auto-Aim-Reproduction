from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np


@dataclass
class RectFeature:
    contour: np.ndarray
    approx: np.ndarray
    bbox: tuple[int, int, int, int]
    area: float
    center: tuple[float, float]


@dataclass
class DetectionResult:
    binary: np.ndarray
    canny: np.ndarray
    fused: np.ndarray
    eroded: np.ndarray
    all_contours: list
    rect_stage_contours: dict
    potential_rects: list
    matched_pairs: list
    filtered_pairs: list
    matched_rects: list
    params: dict


class TargetDetector:
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
    ):
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

    def preprocess(self, frame):
        blurred = cv2.GaussianBlur(frame, (5, 5), self.blur_sigma)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(gray, self.threshold_value, 255, cv2.THRESH_BINARY)[1]
        canny = cv2.Canny(gray, self.canny_low, self.canny_high)
        fused = cv2.bitwise_or(binary, canny)
        eroded=cv2.erode(fused, self.kernel, iterations=1)
        return binary, canny, fused, eroded

    def extract_potential_rects(self, contours):
        potential_rects = []
        stage_contours = {
            "all": [],
            "rejected_area": [],
            "passed_area": [],
            "rejected_vertices": [],
            "passed_vertices": [],
            "rejected_bbox": [],
            "final": [],
        }

        for contour in contours:
            stage_contours["all"].append(contour)
            area = cv2.contourArea(contour)
            if area < self.min_area:
                stage_contours["rejected_area"].append(contour)
                continue
            stage_contours["passed_area"].append(contour)

            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) != 4:
                stage_contours["rejected_vertices"].append(contour)
                continue
            stage_contours["passed_vertices"].append(contour)

            x, y, w, h = cv2.boundingRect(approx)
            if w == 0 or h == 0 or w < h:
                stage_contours["rejected_bbox"].append(contour)
                continue

            center_x = x + w / 2
            center_y = y + h / 2
            rect_feature = RectFeature(
                contour=contour,
                approx=approx,
                bbox=(x, y, w, h),
                area=area,
                center=(center_x, center_y),
            )
            potential_rects.append(rect_feature)
            stage_contours["final"].append(contour)

        return potential_rects, stage_contours

    def match_rects(self, potential_rects):
        matched_pairs = []

        for left, right in combinations(potential_rects, 2):
            x1, y1, w1, h1 = left.bbox
            x2, y2, w2, h2 = right.bbox
            cx1, cy1 = left.center
            cx2, cy2 = right.center

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
                matched_pairs.append((left, right))

        return matched_pairs

    def _unique_contour_count(self, rect_pairs):
        unique_ids = set()
        for left, right in rect_pairs:
            unique_ids.add(id(left.contour))
            unique_ids.add(id(right.contour))
        return len(unique_ids)

    def _flatten_pairs(self, rect_pairs):
        contours = []
        for left, right in rect_pairs:
            contours.extend((left.contour, right.contour))
        return contours

    def _order_points(self, pts):
        ordered = np.zeros((4, 2), dtype=np.float32)
        sums = pts.sum(axis=1)
        diffs = np.diff(pts, axis=1).reshape(-1)
        ordered[0] = pts[np.argmin(sums)]
        ordered[2] = pts[np.argmax(sums)]
        ordered[1] = pts[np.argmin(diffs)]
        ordered[3] = pts[np.argmax(diffs)]
        return ordered

    def _warp_to_a4(self, frame, rect_feature):
        pts = rect_feature.approx.reshape(4, 2).astype(np.float32)
        if pts.shape != (4, 2):
            return None

        ordered = self._order_points(pts)
        width, height = self.a4_size
        dst = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(ordered, dst)
        return cv2.warpPerspective(frame, matrix, (width, height))

    def _detect_circles(self, warped):
        if warped is None:
            return None

        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.2)
        min_dim = min(blurred.shape[:2])
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=self.circle_dp,
            minDist=max(20, min_dim // 4),
            param1=self.circle_param1,
            param2=self.circle_param2,
            minRadius=max(8, min_dim // 18),
            maxRadius=max(16, min_dim // 3),
        )
        return circles

    def _filter_pairs_by_confidence(self, matched_pairs, frame):
        evaluations = []
        valid_pairs = []

        for index, pair in enumerate(matched_pairs, start=1):
            reference_rect = min(pair, key=lambda rect: rect.area)
            warped = self._warp_to_a4(frame, reference_rect)
            circles = self._detect_circles(warped)
            circle_count = 0 if circles is None else circles.shape[1]
            evaluations.append(
                {
                    "pair_index": index,
                    "circle_count": circle_count,
                    "reference_area": round(reference_rect.area, 2),
                }
            )
            if circle_count > 0:
                valid_pairs.append((pair, circle_count, reference_rect.area))

        if not valid_pairs:
            return [], evaluations

        if len(valid_pairs) == 1:
            return [valid_pairs[0][0]], evaluations

        best_pair = max(valid_pairs, key=lambda item: (item[1], item[2]))
        return [best_pair[0]], evaluations

    def detect(self, frame):
        binary, canny, fused, eroded = self.preprocess(frame)
        all_contours, _ = cv2.findContours(eroded, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        potential_rect_features, rect_stage_contours = self.extract_potential_rects(all_contours)
        matched_pairs = self.match_rects(potential_rect_features)

        filtered_pairs = matched_pairs
        pair_evaluations = []
        if self._unique_contour_count(matched_pairs) > 2:
            filtered_pairs, pair_evaluations = self._filter_pairs_by_confidence(matched_pairs, frame)

        matched_rects = self._flatten_pairs(filtered_pairs)
        params = {
            "threshold": self.threshold_value,
            "min_area": self.min_area,
            "kernel": f"{self.kernel_size[0]}x{self.kernel_size[1]}",
            "canny": f"{self.canny_low}/{self.canny_high}",
            "raw_contours": len(all_contours),
            "reject_area": len(rect_stage_contours["rejected_area"]),
            "pass_area": len(rect_stage_contours["passed_area"]),
            "reject_vertices": len(rect_stage_contours["rejected_vertices"]),
            "pass_vertices": len(rect_stage_contours["passed_vertices"]),
            "reject_bbox": len(rect_stage_contours["rejected_bbox"]),
            "candidate_rects": len(potential_rect_features),
            "matched_pairs": len(matched_pairs),
            "filtered_pairs": len(filtered_pairs),
            "filtered_rects": len(matched_rects),
        }
        if pair_evaluations:
            params["pair_circles"] = ",".join(str(item["circle_count"]) for item in pair_evaluations)

        return DetectionResult(
            binary=binary,
            canny=canny,
            fused=fused,
            eroded=eroded,
            all_contours=all_contours,
            rect_stage_contours=rect_stage_contours,
            potential_rects=[feature.contour for feature in potential_rect_features],
            matched_pairs=[(left.contour, right.contour) for left, right in matched_pairs],
            filtered_pairs=[(left.contour, right.contour) for left, right in filtered_pairs],
            matched_rects=matched_rects,
            params=params,
        )
