from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np


@dataclass
class DetectionResult:
    binary: np.ndarray
    closed: np.ndarray
    potential_rects: list
    matched_rects: list


class Detector:
    def __init__(self, min_area=1500, threshold_value=120, kernel_size=(5, 5)):
        self.min_area = min_area
        self.threshold_value = threshold_value
        self.kernel = np.ones(kernel_size, np.uint8)

    def preprocess(self, frame):
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(gray, self.threshold_value, 255, cv2.THRESH_BINARY)[1]
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.kernel)
        return binary, closed

    def extract_potential_rects(self, closed_mask):
        contours, _ = cv2.findContours(closed_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        potential_rects = []

        for contour in contours:
            if cv2.contourArea(contour) < self.min_area:
                continue

            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) != 4:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w == 0 or h == 0:
                continue
            if w < h:
                continue

            potential_rects.append(contour)

        return potential_rects

    def match_rects(self, potential_rects):
        rect_features = []
        for contour in potential_rects:
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w / 2
            center_y = y + h / 2
            rect_features.append((contour, x, w, h, center_x, center_y))

        matched_indexes = set()
        for (i, (_, x1, w1, h1, cx1, cy1)), (j, (_, x2, w2, h2, cx2, cy2)) in combinations(
            enumerate(rect_features), 2
        ):
            x_limit = max(w1, w2) / 10
            y_limit = max(h1, h2) / 10
            x_gap_limit = max(w1, w2) / 5
            if abs(cx1 - cx2) < x_limit and abs(cy1 - cy2) < y_limit and abs(x1 - x2) <= x_gap_limit:
                matched_indexes.update({i, j})

        return [rect_features[i][0] for i in sorted(matched_indexes)]

    def detect(self, frame):
        binary, closed = self.preprocess(frame)
        potential_rects = self.extract_potential_rects(closed)
        matched_rects = self.match_rects(potential_rects)
        return DetectionResult(
            binary=binary,
            closed=closed,
            potential_rects=potential_rects,
            matched_rects=matched_rects,
        )
