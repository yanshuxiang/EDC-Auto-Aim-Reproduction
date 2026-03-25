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
        """
        初始化结果保存器，统一管理调试图片目录和可选的视频输出。

        参数说明：
        - output_video_path: 调试渲染视频的输出路径；传入 None 时不创建视频写入器。
        - wrong_dir: 保存“识别错误但带完整调试可视化”的图片目录。
        - wrong_pure_dir: 保存“识别错误的原始帧”的图片目录。
        - normal_dir: 保存普通调试帧或原始帧的目录。
        - base_dir: 所有相对路径的基准目录；为空时默认使用当前模块上一级目录。
        - fps: 输出视频帧率。
        - frame_size: 输出视频尺寸，需与写入的视频帧尺寸一致。

        初始化流程：
        1. 解析并保存所有目录/文件路径。
        2. 重建调试输出目录，避免旧结果干扰当前运行。
        3. 如配置了视频输出，则创建 OpenCV 的 VideoWriter。
        """
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
        """
        将传入路径规范化为绝对路径字符串。

        处理规则：
        - 如果 path 已经是绝对路径，则直接返回。
        - 如果 path 是相对路径，则基于 self.base_dir 拼接并解析成绝对路径。

        这样可以保证后续文件读写都使用统一的、明确的目标位置。
        """
        path_obj = Path(path)
        if path_obj.is_absolute():
            return str(path_obj)
        return str((self.base_dir / path_obj).resolve())

    def _prepare_dirs(self):
        """
        重建调试输出目录。

        每次实例化时都会删除并重新创建 wrong_dir、wrong_pure_dir、normal_dir，
        目的是清空上一次运行遗留的图片，确保当前调试结果是干净且连续编号的。
        """
        for target_dir in (self.wrong_dir, self.wrong_pure_dir, self.normal_dir):
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            os.makedirs(target_dir)

    def _to_bgr(self, image):
        """
        将输入图像转换为适合拼图显示的 BGR 三通道格式。

        OpenCV 的绘图、拼接和文字叠加通常以三通道 BGR 图像最方便：
        - 若输入是单通道灰度图，则转换为 BGR。
        - 若输入本身已经是彩色图，则返回其副本，避免原地修改影响上游数据。
        """
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image.copy()

    def _draw_title(self, image, title):
        """
        在图像左上角绘制标题文字，并返回该图像对象。

        这里直接在传入的 image 上进行原地绘制，调用方如果不希望污染原图，
        应在调用前自行 copy。
        """
        cv2.putText(image, title, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        return image

    def _draw_params(self, image, params):
        """
        将参数字典逐行绘制到图像上，用于记录当前帧处理时的配置值。

        参数按照字典迭代顺序从上到下排布，便于在调试图中同时查看：
        - 当前原始画面
        - 本帧使用的阈值或其他处理参数
        """
        y = 68
        for key, value in params.items():
            line = f"{key}: {value}"
            cv2.putText(image, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            y += 28
        return image

    def _build_contour_stage_view(self, frame, contours, title, color):
        """
        构建某一轮轮廓筛选阶段的可视化图。

        参数说明：
        - frame: 用作底图的原始帧。
        - contours: 当前阶段需要展示的轮廓集合。
        - title: 当前阶段标题，会附带轮廓数量一起显示。
        - color: 绘制轮廓时使用的 BGR 颜色。

        返回值：
        - 一张新的图像副本，其中叠加了指定轮廓和标题信息。
        """
        view = frame.copy()
        if contours:
            cv2.drawContours(view, contours, -1, color, 2)
        self._draw_title(view, f"{title}: {len(contours)}")
        return view

    def _build_ordered_strip(self, views):
        """
        将多张阶段图按从上到下的顺序拼接成一张长图。

        具体处理：
        - 先计算所有子图中的最大宽度。
        - 对宽度不足的图像在右侧补黑边，使所有图宽度一致。
        - 在相邻阶段之间插入一个带“v”标记的分隔条，强调处理流程向下推进。

        返回值：
        - 由所有阶段图垂直拼接得到的总调试图。
        """
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
        """
        根据单帧处理结果构建完整的调试画布。

        这张画布按处理流程依次展示：
        1. 原始图与参数
        2. 二值化 / Canny / 腐蚀等预处理结果
        3. 全部轮廓
        4. 矩形筛选流程中每一步被拒绝或保留的轮廓
        5. 最终匹配出的目标轮廓

        依赖 result 中的多个中间字段，因此该方法本质上是将算法 pipeline
        的中间状态结构化地“展开”为一张便于人工排查的问题定位图。
        """
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
        """
        保存一帧“错误样本”。

        会同时输出两张图：
        - wrong_dir 中保存完整调试拼图，便于分析算法每一步为何失败。
        - wrong_pure_dir 中保存原始帧，便于后续做数据回放、标注或复现。

        wrong_count 只在错误样本维度上单独计数，避免与普通保存帧混淆。
        """
        self.wrong_count += 1
        wrong_path = os.path.join(self.wrong_dir, f"wrong{self.wrong_count}.jpg")
        wrong_pure_path = os.path.join(self.wrong_pure_dir, f"wrong_frame{self.wrong_count}.jpg")
        cv2.imwrite(wrong_path, self.build_debug_canvas(frame, result))
        cv2.imwrite(wrong_pure_path, frame)

    def save_debug_frame(self, frame, result):
        """
        保存普通调试帧，输出内容为完整调试拼图。

        与 save_wrong_frame 的区别在于：
        - 这里只保存一张调试画布；
        - 使用 save_count 计数，通常用于连续抽样记录正常处理过程。
        """
        self.save_count += 1
        path = os.path.join(self.normal_dir, f"normal{self.save_count}.jpg")
        cv2.imwrite(path, self.build_debug_canvas(frame, result))

    def save_frame(self, frame):
        """
        直接保存原始帧，不叠加任何调试信息。

        适合在只关心输入画面本身、暂时不需要中间处理细节时使用。
        该方法与 save_debug_frame 共用 save_count 计数器。
        """
        self.save_count += 1
        path = os.path.join(self.normal_dir, f"frame{self.save_count}.jpg")
        cv2.imwrite(path, frame)

    def write_render_frame(self, frame):
        """
        将一帧写入输出视频。

        只有在初始化时成功创建了 VideoWriter 的情况下才会实际写入；
        如果未配置 output_video_path，则该方法静默跳过。
        """
        if self.writer is not None:
            self.writer.write(frame)

    def release(self):
        """
        释放视频写入器占用的系统资源。

        在处理结束后应调用该方法，确保视频文件句柄被正确关闭，
        避免输出文件损坏或最后几帧未刷新到磁盘。
        """
        if self.writer is not None:
            self.writer.release()
