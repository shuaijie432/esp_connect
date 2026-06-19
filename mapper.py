"""激光雷达建图器 — 自然闭环边界"""

import math
import time
import threading
from collections import deque
from typing import List, Tuple, Optional

from lidar_parser import parse_frame, LidarPoint, OdomData
from occupancy_grid import OccupancyGridMap
import os


class RobotPose:
    LIDAR_OFFSET_X = 0.0
    LIDAR_OFFSET_Y = 0.0
    LIDAR_OFFSET_THETA = 0.0

    def __init__(self, x: float = 0, y: float = 0, theta: float = 0):
        self.x = x
        self.y = y
        self.theta = theta

    def transform_point(self, local_x: float, local_y: float):
        lx = local_x + self.LIDAR_OFFSET_X
        ly = local_y + self.LIDAR_OFFSET_Y
        cos_t = math.cos(self.theta + self.LIDAR_OFFSET_THETA)
        sin_t = math.sin(self.theta + self.LIDAR_OFFSET_THETA)
        wx = self.x + lx * cos_t - ly * sin_t
        wy = self.y + lx * sin_t + ly * cos_t
        return wx, wy

    def set_pose(self, x: float, y: float, theta: float):
        self.x = x
        self.y = y
        self.theta = theta


class LidarMapper:
    def __init__(self,
                 map_size_mm: int = 8000,
                 resolution_mm: int = 20,
                 pose: Optional[RobotPose] = None,
                 load_map_path: Optional[str] = None):

        # ===== 只保留这一段初始化，删除后面重复的部分 =====
        self.map = OccupancyGridMap(map_size_mm, resolution_mm)
        self.pose = pose or RobotPose()
        self.lock = threading.Lock()

        # === 静态地图模式 ===
        self.static_map_mode = False
        self._load_map_path = load_map_path

        if load_map_path and os.path.exists(load_map_path):
            self.map.load_from_image(load_map_path)
            self.static_map_mode = True
            # 保存原始地图备份，用于 clear_map 时恢复（而非从可能被污染的文件重载）
            self._clean_log_odds = self.map.log_odds.copy()
            print(f"[MAPPER] 静态地图模式: 已加载 {load_map_path}")
        elif load_map_path:
            print(f"[WARN] 地图文件不存在: {load_map_path}")
            self._clean_log_odds = None

        self.frame_count = 0
        self.total_points = 0
        self.latest_points_local = []
        self.latest_points_world = []
        self.all_scanned_points = deque(maxlen=2000)
        self.trajectory = deque(maxlen=2000)
        self.trajectory.append((self.pose.x, self.pose.y))

        self.last_v = 0.0
        self.last_w = 0.0
        self.last_odom_time = time.time()

        # 闭环检测
        self.loop_closure_enabled = True
        self.loop_threshold_mm = 300
        self.loop_theta_threshold = 15
        self.start_pose = (0.0, 0.0, 0.0)
        self.is_first_odom = True
        self.loop_detected = False
        self.loop_correction = (0.0, 0.0, 0.0)

    def update_odom_direct(self, odom: OdomData, timestamp_ms: int):
        if not odom.valid:
            return

        with self.lock:
            if self.is_first_odom:
                self.start_pose = (odom.x, odom.y, odom.theta)
                self.is_first_odom = False

            corrected_x = odom.x + self.loop_correction[0]
            corrected_y = odom.y + self.loop_correction[1]
            corrected_theta = odom.theta + self.loop_correction[2]

            if self.loop_closure_enabled and not self.loop_detected:
                dx = corrected_x - self.start_pose[0]
                dy = corrected_y - self.start_pose[1]
                dist_to_start = math.hypot(dx, dy)
                dtheta = abs(math.degrees(corrected_theta - self.start_pose[2]))

                if dist_to_start < self.loop_threshold_mm and dtheta < self.loop_theta_threshold:
                    if self.frame_count > 100:
                        print(f"[LOOP] 闭环检测！距离起点 {dist_to_start:.0f}mm, 角度差 {dtheta:.1f}°")
                        self.loop_correction = (
                            self.start_pose[0] - odom.x,
                            self.start_pose[1] - odom.y,
                            self._normalize_angle(self.start_pose[2] - odom.theta)
                        )
                        self.loop_detected = True
                        print(f"[LOOP] 应用修正: dx={self.loop_correction[0]:.1f}, "
                              f"dy={self.loop_correction[1]:.1f}, "
                              f"dθ={math.degrees(self.loop_correction[2]):.1f}°")
                        corrected_x = self.start_pose[0]
                        corrected_y = self.start_pose[1]
                        corrected_theta = self.start_pose[2]

            self.last_v = math.hypot(odom.vx, odom.vy)
            self.last_w = odom.wz
            self.pose.set_pose(corrected_x, corrected_y, corrected_theta)

            if len(self.trajectory) > 0:
                last_x, last_y = self.trajectory[-1]
                dist = math.hypot(corrected_x - last_x, corrected_y - last_y)
            else:
                dist = float('inf')

            if dist > 15:
                self.trajectory.append((corrected_x, corrected_y))

            self.last_odom_time = time.time()

    def _normalize_angle(self, angle: float) -> float:
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def get_map_display(self):
        with self.lock:
            return self.map.get_display()

    def get_latest_points(self):
        with self.lock:
            return list(self.latest_points_local), list(self.latest_points_world)

    def get_all_scanned_points(self):
        with self.lock:
            return list(self.all_scanned_points)

    def get_trajectory(self):
        with self.lock:
            return list(self.trajectory)

    def get_stats(self):
        with self.lock:
            return {
                'frame_count': self.frame_count,
                'total_points': self.total_points,
                'pose': (self.pose.x, self.pose.y, self.pose.theta),
                'map_size': self.map.size,
                'history_points': len(self.all_scanned_points),
                'loop_detected': self.loop_detected
            }

    def get_odom_stats(self):
        with self.lock:
            return {
                'x': self.pose.x,
                'y': self.pose.y,
                'theta': self.pose.theta,
                'trajectory_len': len(self.trajectory),
                'last_v': self.last_v,
                'last_w': self.last_w,
                'last_update': time.time() - self.last_odom_time,
                'loop_correction': self.loop_correction
            }

    def save_map(self, filepath: str):
        with self.lock:
            self.map.save(filepath)

    def clear_map(self):
        with self.lock:
            if self.static_map_mode and hasattr(self, '_clean_log_odds') and self._clean_log_odds is not None:
                # 从原始备份恢复，而非从可能被污染的文件重载
                self.map.log_odds = self._clean_log_odds.copy()
                print("[MAPPER] 地图已从原始备份恢复")
            else:
                self.map.log_odds.fill(0)

            self.all_scanned_points.clear()
            self.trajectory.clear()
            self.frame_count = 0
            self.total_points = 0
            self.pose.set_pose(0, 0, 0)
            self.trajectory.append((0, 0))
            self.last_odom_time = time.time()
            self.last_v = 0.0
            self.last_w = 0.0
            self.is_first_odom = True
            self.loop_detected = False
            self.loop_correction = (0.0, 0.0, 0.0)

    def reset_odometry(self):
        with self.lock:
            self.pose.set_pose(0, 0, 0)
            self.trajectory.clear()
            self.trajectory.append((0, 0))
            self.last_odom_time = time.time()
            self.last_v = 0.0
            self.last_w = 0.0
            self.is_first_odom = True
            self.loop_detected = False
            self.loop_correction = (0.0, 0.0, 0.0)

    def filter_points(self, points: list) -> list:
        """严格过滤：只保留可靠边界点，从源头减少黑点"""
        if len(points) < 5:
            return points

        points_sorted = sorted(points, key=lambda p: p.angle)
        filtered = []

        for i, pt in enumerate(points_sorted):
            # 严格距离范围
            if pt.distance < 200:
                continue
            if pt.distance > 3500:
                continue
            if pt.quality < 15:
                continue

            left = points_sorted[i - 1] if i > 0 else None
            right = points_sorted[i + 1] if i < len(points_sorted) - 1 else None

            diff_left = abs(pt.distance - left.distance) if left else 9999
            diff_right = abs(pt.distance - right.distance) if right else 9999

            # 孤立点直接丢弃（噪点主要来源）
            if diff_left > 400 and diff_right > 400:
                continue

            filtered.append(pt)

        return filtered