import cv2
import numpy as np

class KalmanFilterTracker:
    """
    Kalman Filter for 2D target tracking with prediction support.
    用于 2D 目标追踪的卡尔曼滤波器，支持目标丢失后的线性预测。
    """
    def __init__(self, dt=0.01, lost_threshold=10, process_noise=1e-2, measure_noise=1e-1):
        """
        :param dt: Time interval between frames. 帧间时间间隔。
        :param lost_threshold: Max frames to predict after losing target. 丢失目标后允许预测的最大帧数。
        :param process_noise: Process noise covariance (Q). 过程噪声。
        :param measure_noise: Measurement noise covariance (R). 测量噪声。
        """
        # 4 states: [x, y, vx, vy], 2 measurements: [x, y]
        self.kf = cv2.KalmanFilter(4, 2)
        
        # Transition Matrix (F): State transition model
        # x_new = x + vx*dt
        # y_new = y + vy*dt
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)
        
        # Measurement Matrix (H): Observation model
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)
        
        # Process Noise Covariance (Q): Uncertainty in the model
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        
        # Measurement Noise Covariance (R): Uncertainty in sensors/detection
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measure_noise
        
        # Error Covariance (P): Initial uncertainty
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

        self.lost_frames = 0
        self.lost_threshold = lost_threshold
        self.is_initialized = False

    def update(self, pos):
        """
        Update tracker with new position measurement.
        使用新检测到的位置更新追踪器。
        
        :param pos: (x, y) tuple or None if detection failed.
        :return: (filtered_pos, is_tracking_valid)
        """
        if pos is not None:
            measured = np.array([[np.float32(pos[0])], [np.float32(pos[1])]])
            
            if not self.is_initialized:
                # First detection: Initialize state with current position and zero velocity
                self.kf.statePost = np.array([
                    [measured[0, 0]], 
                    [measured[1, 0]], 
                    [0], 
                    [0]
                ], np.float32)
                self.is_initialized = True
                self.lost_frames = 0
                return pos, True
            
            # Predict then correct
            self.kf.predict()
            estimate = self.kf.correct(measured)
            self.lost_frames = 0
            
            filtered_pos = (int(estimate[0, 0]), int(estimate[1, 0]))
            return filtered_pos, True
        
        else:
            # Target lost: Try to predict based on velocity
            if self.is_initialized:
                self.lost_frames += 1
                # Only prediction step
                estimate = self.kf.predict()
                
                if self.lost_frames <= self.lost_threshold:
                    # Still in prediction "grace period"
                    predicted_pos = (int(estimate[0, 0]), int(estimate[1, 0]))
                    return predicted_pos, True
                else:
                    # Exceeded threshold: Truly lost
                    self.is_initialized = False
            
            return None, False

    def reset(self):
        self.is_initialized = False
        self.lost_frames = 0
