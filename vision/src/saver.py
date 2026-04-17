import cv2
import os
from datetime import datetime


class ResultSaver:
    def __init__(self):
        self.output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "debug", "records"))
        os.makedirs(self.output_dir, exist_ok=True)
    
    def create_recorder(self, size, fps):
        """创建视频记录器"""
        # 使用MJPG编码以提高兼容性
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        
        # 限制帧率不超过60，避免MPEG-4编码问题
        limited_fps = min(fps, 60.0)
        
        # 创建输出文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_dir, f"record_{timestamp}.avi")
        
        # 创建VideoWriter
        try:
            writer = cv2.VideoWriter(output_path, fourcc, limited_fps, size)
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open VideoWriter: {output_path}")
            return writer
        except Exception as e:
            raise RuntimeError(f"Failed to open VideoWriter: {output_path} - {str(e)}")
    
    def write(self, frame):
        """写入一帧到视频"""
        if hasattr(self, '_writer') and self._writer is not None:
            try:
                self._writer.write(frame)
            except Exception as e:
                print(f"Error writing frame: {e}")
                self._writer.release()
                self._writer = None
    
    def _open_writer(self, frame):
        """内部方法：打开视频写入器"""
        if not hasattr(self, '_writer') or self._writer is None:
            # 获取尺寸和FPS
            h, w = frame.shape[:2]
            fps = 30.0  # 默认FPS
            
            # 创建新的写入器
            self._writer = self.create_recorder((w, h), fps)
    
    def release(self):
        """释放资源"""
        if hasattr(self, '_writer') and self._writer is not None:
            self._writer.release()
            self._writer = None