import time

class PID:
    """
    A simple and robust PID controller implementation.
    一个简单且稳健的 PID 控制器实现。
    """
    def __init__(self, kp, ki, kd, output_limit=None, integral_limit=None):
        """
        Initialize the PID controller.
        初始化 PID 控制器。

        :param kp: Proportional gain (比例增益)
        :param ki: Integral gain (积分增益)
        :param kd: Derivative gain (微分增益)
        :param output_limit: Maximum absolute value for the output (输出限幅)
        :param integral_limit: Maximum absolute value for the integral term (积分限幅/抗饱和)
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self.output_limit = output_limit
        self.integral_limit = integral_limit
        
        self.prev_error = 0
        self.integral = 0
        self.prev_time = time.time()

    def update(self, error, dt=None):
        """
        Calculate PID output based on current error.
        根据当前误差计算 PID 输出。

        :param error: Current error (setpoint - measure) (当前误差)
        :param dt: Time interval since last update. If None, it calculates automatically. (采样时间间隔)
        :return: Controller output (控制器输出)
        """
        current_time = time.time()
        if dt is None:
            dt = current_time - self.prev_time
        
        # Avoid division by zero
        if dt <= 0:
            dt = 1e-6

        # Proportional term (P项)
        p_out = self.kp * error
        
        # Integral term (I项)
        self.integral += error * dt
        # Anti-windup: Limit the integral term (抗饱和：限制积分项)
        if self.integral_limit is not None:
            if self.integral > self.integral_limit:
                self.integral = self.integral_limit
            elif self.integral < -self.integral_limit:
                self.integral = -self.integral_limit
        i_out = self.ki * self.integral
        
        # Derivative term (D项)
        derivative = (error - self.prev_error) / dt
        d_out = self.kd * derivative
        
        # Calculate total output
        output = p_out + i_out + d_out
        
        # Output saturation: Limit the final output (输出饱和：限制最终输出)
        if self.output_limit is not None:
            if output > self.output_limit:
                output = self.output_limit
            elif output < -self.output_limit:
                output = -self.output_limit
            
        # Save state for next update
        self.prev_error = error
        self.prev_time = current_time
        
        return output

    def reset(self):
        """
        Reset the internal state of the controller.
        重置控制器的内部状态。
        """
        self.prev_error = 0
        self.integral = 0
        self.prev_time = time.time()

if __name__ == "__main__":
    # Example usage:
    # 示例用法：
    pid = PID(kp=1.0, ki=0.1, kd=0.05, output_limit=100, integral_limit=50)
    
    target = 10.0
    current_value = 0.0
    
    print(f"Target: {target}")
    for i in range(20):
        error = target - current_value
        output = pid.update(error, dt=0.1)
        current_value += output * 0.1  # Simplified plant simulation
        print(f"Step {i+1}: Output={output:6.2f}, Current Value={current_value:6.2f}")
