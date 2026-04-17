import numpy as np

def order_points(pts):
    """
    将 4 个顶点排序为：左上、右上、右下、左下
    顺序说明：
    0: top-left, 1: top-right, 2: bottom-right, 3: bottom-left
    """
    # 确保输入是 (4, 2) 的 numpy 数组
    pts = np.array(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")

    # 计算每个点的坐标和 (x + y)
    # 左上角的和最小，右下角的和最大
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # 计算点之间的差 (y - x)
    # 右上角的差最小，左下角的差最大
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def get_distance(pt1, pt2):
    """计算两点间的欧几里得距离（常用于过滤或校准）"""
    return np.sqrt(((pt1[0] - pt2[0]) ** 2) + ((pt1[1] - pt2[1]) ** 2))