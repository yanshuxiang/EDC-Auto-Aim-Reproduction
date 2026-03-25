import os
import shutil
from pathlib import Path

import cv2
import numpy as np


class ResultSaver:
    def __init__(
        self,
        output_video_path="output.mp4",
        wrong_dir="debug/wrong_pic",
        wrong_pure_dir="debug/wrong_pic_pure",
        normal_dir="debug/normal",
        base_dir=None,
        fps=30.0,
        frame_size=(1280, 720),
    ):
        self.base_dir = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent.parent
        self.output_video_path = self._resolve_path(output_video_path) if output_video_path else None
        self.wrong_dir = self._resolve_path(wrong_dir)
        self.wrong_pure_dir = self._resolve_path(wrong_pure_dir)
        self.normal_dir = self._resolve_path(normal_dir)
        self.fps = fps
        self.frame_size = frame_size
        self.wrong_count = 0
        self.save_count = 0

        self._prepare_dirs()
        self.writer = None
        if self.output_video_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(self.output_video_path, fourcc, self.fps, self.frame_size)
            if not self.writer.isOpened():
                raise RuntimeError(f"Failed to open VideoWriter: {self.output_video_path}")

    def _resolve_path(self, path):
        path_obj = Path(path)
        if path_obj.is_absolute():
            return str(path_obj)
        return str((self.base_dir / path_obj).resolve())

    def _prepare_dirs(self):
        for target_dir in (self.wrong_dir, self.wrong_pure_dir, self.normal_dir):
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            os.makedirs(target_dir)

    def _to_bgr(self, image):
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image.copy()

    def _draw_title(self, image, title):
        cv2.putText(image, title, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        return image

    def _draw_params(self, image, params):
        y = 68
        for key, value in params.items():
            line = f"{key}: {value}"
            cv2.putText(image, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            y += 28
        return image

    def _build_contour_stage_view(self, frame, contours, title, color):
        view = frame.copy()
        if contours:
            cv2.drawContours(view, contours, -1, color, 2)
        self._draw_title(view, f"{title}: {len(contours)}")
        return view

    def _build_ordered_strip(self, views):
        separator_height = 20
        strips = []
        width = max(view.shape[1] for view in views)
        for index, view in enumerate(views):
            padded_view = view
            if view.shape[1] < width:
                pad_width = width - view.shape[1]
                padded_view = cv2.copyMakeBorder(
                    view,
                    0,
                    0,
                    0,
                    pad_width,
                    cv2.BORDER_CONSTANT,
                    value=(0, 0, 0),
                )
            strips.append(padded_view)
            if index == len(views) - 1:
                continue
            separator = np.full((separator_height, width, 3), 32, dtype=np.uint8)
            cv2.putText(
                separator,
                "v",
                (width // 2 - 8, 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            strips.append(separator)
        return np.vstack(strips)

    def build_debug_canvas(self, frame, result):
        original_view = frame.copy()
        self._draw_title(original_view, "Step 0 Original + Params")
        self._draw_params(original_view, result.params)

        binary_view = self._draw_title(self._to_bgr(result.binary), "Binary")
        canny_view = self._draw_title(self._to_bgr(result.canny), "Canny")
        eroded_view = self._draw_title(self._to_bgr(result.eroded), "Eroded")
        preprocess_view = np.hstack((binary_view, canny_view, eroded_view))
        self._draw_title(preprocess_view, "Step 1 Preprocess")

        all_contours_view = frame.copy()
        cv2.drawContours(all_contours_view, result.all_contours, -1, (0, 0, 255), 1)
        self._draw_title(all_contours_view, f"Step 2 All Contours: {len(result.all_contours)}")

        ordered_views = [
            original_view,
            preprocess_view,
            all_contours_view,
            self._build_contour_stage_view(
                frame,
                result.rect_stage_contours.get("rejected_area", []),
                "Step 3 Reject Area",
                (0, 0, 255),
            ),
            self._build_contour_stage_view(
                frame,
                result.rect_stage_contours.get("passed_area", []),
                "Step 4 Pass Area",
                (0, 255, 0),
            ),
            self._build_contour_stage_view(
                frame,
                result.rect_stage_contours.get("rejected_vertices", []),
                "Step 5 Reject Vertices!=4",
                (0, 0, 255),
            ),
            self._build_contour_stage_view(
                frame,
                result.rect_stage_contours.get("passed_vertices", []),
                "Step 6 Pass Vertices=4",
                (0, 255, 0),
            ),
            self._build_contour_stage_view(
                frame,
                result.rect_stage_contours.get("rejected_bbox", []),
                "Step 7 Reject BBox",
                (0, 0, 255),
            ),
            self._build_contour_stage_view(
                frame,
                result.rect_stage_contours.get("final", []),
                "Step 8 Final Rects",
                (0, 255, 255),
            ),
        ]

        filtered_view = frame.copy()
        cv2.drawContours(filtered_view, result.matched_rects, -1, (0, 255, 0), 3)
        self._draw_title(filtered_view, f"Step 9 Filtered Contours: {len(result.matched_rects)}")
        ordered_views.append(filtered_view)

        return self._build_ordered_strip(ordered_views)

    def save_wrong_frame(self, frame, result):
        self.wrong_count += 1
        wrong_path = os.path.join(self.wrong_dir, f"wrong{self.wrong_count}.jpg")
        wrong_pure_path = os.path.join(self.wrong_pure_dir, f"wrong_frame{self.wrong_count}.jpg")
        cv2.imwrite(wrong_path, self.build_debug_canvas(frame, result))
        cv2.imwrite(wrong_pure_path, frame)

    def save_debug_frame(self, frame, result):
        self.save_count += 1
        path = os.path.join(self.normal_dir, f"normal{self.save_count}.jpg")
        cv2.imwrite(path, self.build_debug_canvas(frame, result))

    def save_frame(self, frame):
        self.save_count += 1
        path = os.path.join(self.normal_dir, f"frame{self.save_count}.jpg")
        cv2.imwrite(path, frame)

    def write_render_frame(self, frame):
        if self.writer is not None:
            self.writer.write(frame)

    def release(self):
        if self.writer is not None:
            self.writer.release()
