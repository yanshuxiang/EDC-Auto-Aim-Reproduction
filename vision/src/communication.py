import serial
import struct
import time

class SerialCommunicator:
    """
    Serial communication module for sending target and actual positions to hardware.
    串口通讯模块，按协议向下位机发送目标和实际位置数据。
    """
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200, timeout=0.1):
        """
        :param port: Serial port path (e.g., /dev/ttyUSB0 for Ubuntu).
        :param baudrate: Communication speed.
        :param timeout: Read timeout in seconds.
        """
        try:
            self.ser = serial.Serial(port, baudrate, timeout=timeout)
            print(f"Successfully connected to serial port: {port} at {baudrate} bps")
        except Exception as e:
            self.ser = None
            print(f"Warning: Could not open serial port {port}: {e}")

    def send_data(self, target_pos, actual_pos, type_byte=0x01):
        """
        Package and send coordinate data according to the 11-byte protocol.
        打包并发送坐标数据。协议格式：
        Byte0: 0xAA (Header)
        Byte1: Type/Reserved
        Byte2-3: Target X (H, L)
        Byte4-5: Target Y (H, L)
        Byte6-7: Actual X (H, L)
        Byte8-9: Actual Y (H, L)
        Byte10: 0x55 (Tail)
        """
        if self.ser is None or not self.ser.is_open:
            return False

        try:
            # Map coordinates to signed 16-bit integers
            tx = int(target_pos[0]) if target_pos else 0
            ty = int(target_pos[1]) if target_pos else 0
            ax = int(actual_pos[0]) if actual_pos else 0
            ay = int(actual_pos[1]) if actual_pos else 0

            # Constrain to 16-bit signed range (-32768 to 32767)
            tx = max(min(tx, 32767), -32768)
            ty = max(min(ty, 32767), -32768)
            ax = max(min(ax, 32767), -32768)
            ay = max(min(ay, 32767), -32768)

            # Frame Header and Tail
            header = 0xAA
            tail = 0x55
            
            # Pack data: > (Big Endian), B (Header), B (Type), hhhh (4 signed shorts), B (Tail)
            frame = struct.pack('>BBhhhhB', header, type_byte, tx, ty, ax, ay, tail)
            
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
    comm = SerialCommunicator(port='/dev/ttyUSB0', baudrate=115200)
    if comm.ser:
        while True:
            try:
                # Mock data test
                comm.send_data((320, 240), (310, 230))
                time.sleep(0.1)
            except KeyboardInterrupt:
                break
        comm.close()

