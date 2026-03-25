import cv2

from vision.src.capture import FrameCapture
from vision.src.target_detector import TargetDetector
from vision.src.saver import ResultSaver


class DetectionApp:
    def __init__(
        self,
        source="media/1.mp4",
        output_video="output.mp4",
        min_area=1500,
        threshold_value=120,
        render_fps=60.0,
    ):
        self.source = source
        self.output_video = output_video
        self.target_detector = TargetDetector(min_area=min_area, threshold_value=threshold_value)
        self.render_fps = render_fps

    def run(self):
        with FrameCapture(self.source) as capture:
            saver = ResultSaver(
                output_video_path=self.output_video,
                fps=self.render_fps,
                frame_size=capture.get_size(),
            )
            try:
                while True:
                    ret, frame = capture.read()
                    if not ret:
                        break

                    result = self.target_detector.detect(frame)
                    if len(result.matched_rects) != 2:
                        saver.save_wrong_frame(frame, result)

                    render_frame = frame.copy()
                    cv2.drawContours(render_frame, result.matched_rects, -1, (0, 0, 255), 3)
                    saver.write_render_frame(render_frame)
                    cv2.imshow("frame", render_frame)

                    key = cv2.waitKey(20) & 0xFF
                    if key == ord("s"):
                        saver.save_debug_frame(frame, result)
                    elif key == ord("q"):
                        break
            finally:
                saver.release()
                cv2.destroyAllWindows()


if __name__ == "__main__":
    DetectionApp().run()
