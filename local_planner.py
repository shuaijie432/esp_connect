"""局部代价地图 + 简化 DWA — 快速响应"""

import math
import numpy as np
from typing import List, Tuple, Optional


class LocalCostMap:
    """简化版局部代价地图"""

    def __init__(self, size_mm: int = 3000, resolution_mm: int = 20):
        self.resolution = resolution_mm
        self.size = size_mm // resolution_mm  # 150×150
        self.half_size = self.size // 2

        self.costmap = np.zeros((self.size, self.size), dtype=np.uint8)
        self.timemap = np.zeros((self.size, self.size), dtype=float)
        self.confidence = np.zeros((self.size, self.size), dtype=np.float32)

        self.decay_time = 1.5
        self.robot_radius_cells = 11.5  # 18cm 半径（33cm 正方形的内切圆）
        self.inflation_radius = 2   # 24cm 膨胀

        # 预计算 footprint 掩码
        r = self.robot_radius_cells
        y, x = np.ogrid[-r:r+1, -r:r+1]
        self._footprint_mask = (x**2 + y**2) <= r**2

    def world_to_local(self, wx, wy, robot_x, robot_y, robot_theta):
        dx = wx - robot_x
        dy = wy - robot_y
        local_x = dx * math.cos(robot_theta) + dy * math.sin(robot_theta)
        local_y = -dx * math.sin(robot_theta) + dy * math.cos(robot_theta)
        mx = int(local_x / self.resolution + self.half_size)
        my = int(local_y / self.resolution + self.half_size)
        return mx, my

    def update(self, world_points, robot_x, robot_y, robot_theta, timestamp):
        # 快速衰减
        dt = timestamp - self.timemap
        self.confidence[dt > self.decay_time] *= 0.3
        self.confidence = np.clip(self.confidence, 0, 5)

        # 更新点云
        for wx, wy in world_points:
            mx, my = self.world_to_local(wx, wy, robot_x, robot_y, robot_theta)
            if 0 <= mx < self.size and 0 <= my < self.size:
                self.confidence[my, mx] += 1.0
                self.timemap[my, mx] = timestamp

        # 生成代价地图
        occupied = self.confidence > 1.5
        self.costmap.fill(0)
        self.costmap[occupied] = 100

        # 快速膨胀
        self._fast_inflate()

    def _fast_inflate(self):
        """快速膨胀，不循环"""
        occ = self.costmap > 50
        # 一次膨胀 3 个栅格
        for _ in range(3):
            padded = np.pad(occ, 1, mode='constant', constant_values=False)
            occ |= padded[0:-2, 1:-1] | padded[2:, 1:-1] | padded[1:-1, 0:-2] | padded[1:-1, 2:]
        self.costmap[occ] = 255

    def check_collision(self, local_x_mm, local_y_mm):
        cx = int(local_x_mm / self.resolution + self.half_size)
        cy = int(local_y_mm / self.resolution + self.half_size)

        mask_h, mask_w = self._footprint_mask.shape
        half = mask_h // 2

        for dy in range(mask_h):
            for dx in range(mask_w):
                if not self._footprint_mask[dy, dx]:
                    continue
                mx = cx + dx - half
                my = cy + dy - half
                if 0 <= mx < self.size and 0 <= my < self.size:
                    if self.costmap[my, mx] >= 255:
                        return True
        return False

    def get_cost(self, local_x_mm, local_y_mm):
        mx = int(local_x_mm / self.resolution + self.half_size)
        my = int(local_y_mm / self.resolution + self.half_size)
        if 0 <= mx < self.size and 0 <= my < self.size:
            return self.costmap[my, mx]
        return 255


class DWAPlanner:
    """简化版 DWA — 快速采样"""

    def __init__(self, local_costmap):
        self.local_costmap = local_costmap

        # 速度限制
        self.max_vx = 300.0
        self.max_vy = 200.0
        self.max_vw = 0.5  # 限制角速度！

        # 采样参数：大幅减少
        self.vx_samples = 5   # 只采样 5 个前向速度
        self.vy_samples = 7   # 7 个侧向速度
        self.vw_samples = 3   # 3 个角速度
        self.dt = 0.1
        self.predict_time = 0.4  # 缩短预测时间

    def plan(self, x, y, theta, target, current_vx, current_vy, current_vw, waypoints, current_wp):
        """简化版：直接搜索最优 (vx, vy, vw)"""

        # 固定速度候选，不计算动态窗口
        vx_candidates = [250, 150, 50, 0, -100]  # 前向速度
        vy_candidates = [-200, -150, -100, 0, 100, 150, 200]  # 侧向（负=右）
        vw_candidates = [0.0]  # 禁用旋转！

        best_score = -float('inf')
        best_vx, best_vy, best_vw = 0.0, 0.0, 0.0

        for vx in vx_candidates:
            for vy in vy_candidates:
                for vw in vw_candidates:
                    # 快速碰撞检测
                    if self._check_collision_simple(x, y, theta, vx, vy, vw):
                        continue

                    score = self._evaluate(x, y, theta, vx, vy, vw, target)
                    if score > best_score:
                        best_score = score
                        best_vx, best_vy, best_vw = vx, vy, vw

        if best_score == -float('inf'):
            return 0.0, 0.0, 0.0  # 无可行解

        return best_vx, best_vy, best_vw

    def _check_collision_simple(self, x, y, theta, vx, vy, vw):
        """简化碰撞检测：只检查终点"""
        # 模拟一步
        theta_new = theta + vw * self.dt
        theta_mid = (theta + theta_new) / 2.0

        new_x = x + vx * math.cos(theta_mid) * self.dt - vy * math.sin(theta_mid) * self.dt
        new_y = y + vx * math.sin(theta_mid) * self.dt + vy * math.cos(theta_mid) * self.dt

        # 转到局部坐标
        local_x = (new_x - x) * math.cos(theta) + (new_y - y) * math.sin(theta)
        local_y = -(new_x - x) * math.sin(theta) + (new_y - y) * math.cos(theta)

        return self.local_costmap.check_collision(local_x, local_y)

    def _evaluate(self, x, y, theta, vx, vy, vw, target):
        """简化评分"""
        # 1. 目标方向
        dx = target[0] - x
        dy = target[1] - y
        goal_dist = math.hypot(dx, dy)

        # 2. 避障（检查前方几个点）
        collision_cost = 0
        for step in [1, 2, 3]:
            t = step * self.dt
            theta_new = theta + vw * t
            theta_mid = (theta + theta_new) / 2.0

            check_x = x + vx * math.cos(theta_mid) * t - vy * math.sin(theta_mid) * t
            check_y = y + vx * math.sin(theta_mid) * t + vy * math.cos(theta_mid) * t

            local_x = (check_x - x) * math.cos(theta) + (check_y - y) * math.sin(theta)
            local_y = -(check_x - x) * math.sin(theta) + (check_y - y) * math.cos(theta)

            cost = self.local_costmap.get_cost(local_x, local_y)
            if cost >= 255:
                return -float('inf')  # 碰撞
            collision_cost += cost / 255.0

        # 3. 评分
        # 前进奖励
        forward_score = vx / 300.0

        # 侧向奖励（有障碍时鼓励侧移）
        lateral_score = abs(vy) / 200.0

        # 目标接近奖励
        goal_score = max(0, 1.0 - goal_dist / 2000.0)

        # 避障惩罚
        obstacle_score = 1.0 - min(collision_cost / 3, 1.0)

        score = (0.3 * forward_score +
                 0.3 * lateral_score +
                 0.2 * goal_score +
                 0.2 * obstacle_score)

        return score