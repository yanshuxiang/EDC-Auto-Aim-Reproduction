import sys
import os
import time

# 获取路径信息
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)

# 动态添加路径以寻找通讯模块
sys.path.insert(0, script_dir)
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(script_dir, 'src'))

try:
    # 尝试从 src 直接导入 (如果已经在 vision 目录下)
    from communication import SerialCommunicator
except ImportError:
    try:
        # 尝试按照主程序的方式从 vision.src 导入
        from vision.src.communication import SerialCommunicator
    except ImportError:
        print("错误：无法加载通讯模块。请确认文件路径：vision/src/communication.py")
        sys.exit(1)

def find_usb_serial():
    """尝试自动寻找可用串口"""
    try:
        import serial.tools.list_ports
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            device = p.device.lower()
            if any(key in device for key in ['usb', 'serial', 'ch340', 'cp210', 'ftdi', 'cu.']):
                return p.device
    except ImportError:
        pass
    return None

def test_serial():
    # 自动探测串口，如果没发现则使用默认值
    port = find_usb_serial() or '/dev/ttyUSB0'
    
    print(f"========================================")
    print(f"       串口发送测试程序 (Gimbal Test)    ")
    print(f"========================================")
    print(f"目标：每隔 1 秒增加 5 个单位（即转动 5 度）")
    print(f"尝试连接串口: {port}")
    
    # 初始化通讯模块
    try:
        comm = SerialCommunicator(port=port, baudrate=115200)
    except Exception as e:
        print(f"[错误] 初始化失败: {e}")
        return
    
    if not comm.ser or not comm.ser.is_open:
        print("\n[错误] 无法打开串口。")
        print("1. 检查物理连接。")
        print("2. 检查权限 (Linux: sudo chmod 666 /dev/ttyUSB0)。")
        print("3. 当前尝试端口: " + port)
        return

    angle = 0
    print("\n[成功] 串口已连接。按下 Ctrl+C 停止测试。")
    
    try:
        while True:
            # 假设 target_pos[0] (tx) 为 Yaw 轴角度
            target_val = int(angle)
            
            # 打印当前发送的数据
            print(f"[{time.strftime('%H:%M:%S')}] 发送 -> 目标角度: {target_val:5d}")
            
            # 发送数据包
            # 协议格式：Header(0xAA), Type, TX, TY, AX, AY, Tail(0x55)
            # 这里 tx 设为递增角度，ty/ax/ay 设为 0
            success = comm.send_data(target_pos=(target_val, 0), actual_pos=(0, 0))
            
            if not success:
                print("   !! 发送失败，请检查连接 !!")
            
            # 步进 5 度
            angle += 5
            
            # 循环控制：如果不想角度无限增大，可以加取模运算
            # angle %= 360 
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n测试停止。")
    finally:
        comm.close()
        print("串口已安全关闭。")

if __name__ == "__main__":
    test_serial()
