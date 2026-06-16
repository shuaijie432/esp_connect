# odom.py
import math
import time
from collections import deque


class Odometry:
    """
    接收 v, w，积分得到位姿 x, y, theta
    （当前系统使用 ESP32 直接上传绝对位姿，此模块暂未调用，保留备用）
    """

    def __init__(self, x=0.0, y=0.0, theta=0.0):
        self.x = x  # mm
        self.y = y  # mm
        self.theta = theta  # rad

        self.last_timestamp = None
        self.trajectory = deque(maxlen=10000)
        self.trajectory.append((x, y))

    def update(self, v: float, w: float, timestamp_ms: int):
        """
        更新位姿

        Args:
            v: 线速度 mm/s
            w: 角速度 rad/s
            timestamp_ms: 时间戳 ms（STM32 的相对时间）
        """
        if self.last_timestamp is None:
            self.last_timestamp = timestamp_ms
            return

        dt = (timestamp_ms - self.last_timestamp) / 1000.0  # 转秒
        self.last_timestamp = timestamp_ms

        if dt <= 0 or dt > 1.0:  # 异常时间跳变
            return

        # 中点积分
        theta_new = self.theta + w * dt
        theta_mid = (self.theta + theta_new) / 2.0

        self.x += v * math.cos(theta_mid) * dt
        self.y += v * math.sin(theta_mid) * dt
        self.theta = theta_new

        # 归一化到 [-pi, pi]
        while self.theta > math.pi:
            self.theta -= 2 * math.pi
        while self.theta < -math.pi:
            self.theta += 2 * math.pi

        self.trajectory.append((self.x, self.y))

    def get_pose(self):
        return self.x, self.y, self.theta

    def get_trajectory(self):
        return list(self.trajectory)

    def reset(self, x=0, y=0, theta=0):
        self.x, self.y, self.theta = x, y, theta
        self.trajectory.clear()
        self.trajectory.append((x, y))
        self.last_timestamp = None