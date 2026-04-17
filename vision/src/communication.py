import serial
import struct
import time

class SerialCommunicator:
    """
    Serial communication module for sending control commands to hardware (e.g., STM32/Arduino).
    串口通讯模块，用于向下位机发送控制指令。
    """
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200, timeout=0.1):
        """
        :param port: Serial port path (e.g., /dev/ttyUSB0 for Ubuntu).
        :param baudrate: Communication speed.
        :param timeout: Read timeout in seconds.
        """
        try:
            self.ser = serial.Serial(port, baudrate, timeout=timeout)
            print(f"Successfully connected to serial port: {port}")
        except Exception as e:
            self.ser = None
            print(f"Warning: Could not open serial port {port}: {e}")

    def send_control(self, x_out, y_out):
        """
        Package and send control data.
        打包并发送控制数据。
        
        Frame Format (示例协议):
        [0xAA] [X_high] [X_low] [Y_high] [Y_low] [Checksum] [0x55]
        """
        if self.ser is None or not self.ser.is_open:
            return False

        try:
            # Map float outputs to signed 16-bit integers
            # 将浮点输出映射为 16 位有符号整数
            ix = int(max(min(x_out, 32767), -32768))
            iy = int(max(min(y_out, 32767), -32768))

            # Calculate a simple checksum (sum of all bytes modulo 256)
            # 计算简单校验和
            header = 0xAA
            tail = 0x55
            # Extract high and low bytes for X and Y
            xh, xl = (ix >> 8) & 0xFF, ix & 0xFF
            yh, yl = (iy >> 8) & 0xFF, iy & 0xFF
            
            checksum = (header + xh + xl + yh + yl + tail) & 0xFF

            # Pack data using struct: B=unsigned char, h=signed short
            # 使用 struct 打包数据
            # '<' little endian, 'B' header, 'h' x, 'h' y, 'B' checksum, 'B' tail
            # Note: Protocol above uses high/low byte explicitly, here we match that manual logic
            frame = bytearray([header, xh, xl, yh, yl, checksum, tail])
            
            self.ser.write(frame)
            return True
        except Exception as e:
            print(f"Serial write error: {e}")
            return False

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

if __name__ == "__main__":
    # Test script
    comm = SerialCommunicator(port='/dev/ttyUSB0')
    if comm.ser:
        while True:
            try:
                comm.send_control(100, -100)
                time.sleep(0.1)
            except KeyboardInterrupt:
                break
        comm.close()
