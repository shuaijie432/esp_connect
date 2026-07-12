"""激光里程计 —— 基于粒子滤波（Particle Filter）的定位

核心思想：
- 维护 N 个粒子，每个粒子代表一个位姿假设 (x, y, theta)
- 运动模型：根据轮式里程计增量传播粒子
- 观测模型：将点云投影到地图，计算与障碍物的匹配度作为权重
- 重采样：高权重粒子复制，低权重粒子淘汰
- 输出：所有粒子的加权平均位姿（自然平滑，静止时收敛稳定）

优点：
- 多粒子投票，位姿输出平滑不抖动
- 静止时粒子收敛到真实位姿附近，输出稳定
- 可以处理 kidnapped robot 问题（全局定位）
- 天然支持多假设，比单点穷举搜索鲁棒
"""

import math
import numpy as np
from typing import List, Tuple, Optional


class Particle:
    """单个粒子：位姿 + 权重"""
    __slots__ = ['x', 'y', 'theta', 'weight']

    def __init__(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0, weight: float = 1.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.weight = weight

    def copy(self):
        return Particle(self.x, self.y, self.theta, self.weight)


class LaserOdometry:
    """
    激光里程计 —— 粒子滤波定位
    """

    def __init__(self, num_particles: int = 300, map_grid=None, resolution_mm: int = 15):
        self.map_grid = map_grid
        self.resolution = resolution_mm

        # 粒子群
        self.num_particles = num_particles
        self.particles: List[Particle] = []

        # 输出位姿（粒子加权平均）
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # 位姿协方差（用于判断收敛/发散）
        self.pose_std = (0.0, 0.0, 0.0)

        # 轮式里程计上一次读数（用于计算增量）
        self.last_wheel_x = 0.0
        self.last_wheel_y = 0.0
        self.last_wheel_theta = 0.0
        self.wheel_initialized = False

        # 运动模型噪声（静止时 noise 小，运动时 noise 大）
        self.motion_noise_xy_static = 0.5     # 静止时位置噪声 mm
        self.motion_noise_theta_static = 0.005  # 静止时角度噪声 rad
        self.motion_noise_xy_dynamic = 10.0   # 运动时位置噪声 mm（降低：30→10，减少漂移）
        self.motion_noise_theta_dynamic = 0.03 # 运动时角度噪声 rad（降低：0.15→0.05，减少角漂）

        # 静止判定阈值：轮式增量小于此值认为是噪声，直接忽略
        self.stationary_dx_threshold = 2.0    # mm
        self.stationary_dy_threshold = 2.0    # mm
        self.stationary_dtheta_threshold = 0.02  # rad (~1.1°)

        # 观测模型参数（强化：增大 hit/miss 差距，提高定位校正力）
        self.hit_score = 4.0      # 命中障碍物得分（提高：2.0→3.0）
        self.miss_penalty = -1.0  # 落在空闲区扣分（加强：-0.5→-1.0）
        self.out_penalty = -1.0   # 越界扣分

        # 重采样参数
        self.resample_threshold = 0.5  # 有效粒子数占比低于此值时重采样

        # 统计
        self.frame_count = 0
        self.resample_count = 0

        # 静止检测
        self.stationary_counter = 0
        self.is_stationary = True
        self.stationary_threshold = 3  # 连续3帧位移<5mm认为静止

    def set_map(self, map_grid):
        """绑定参考地图"""
        self.map_grid = map_grid
        print(f"[PF] 地图已绑定: {map_grid.size}x{map_grid.size}, "
              f"分辨率={map_grid.resolution}mm")

    def _init_particles(self, x: float, y: float, theta: float):
        """初始化粒子群（高斯散布）"""
        self.particles = []
        for _ in range(self.num_particles):
            p = Particle(
                x + np.random.normal(0, 50),
                y + np.random.normal(0, 50),
                theta + np.random.normal(0, 0.1),
                1.0 / self.num_particles
            )
            self.particles.append(p)
        print(f"[PF] 粒子初始化完成: {self.num_particles}个, 中心=({x:.0f},{y:.0f}) "
              f"θ={math.degrees(theta):.1f}°")

    def update_wheel_odom(self, odom_x: float, odom_y: float, odom_theta: float):
        """更新轮式里程计（计算增量用于运动模型）"""
        if not self.wheel_initialized:
            self.last_wheel_x = odom_x
            self.last_wheel_y = odom_y
            self.last_wheel_theta = odom_theta
            self.wheel_initialized = True
            # 初始化粒子群
            self._init_particles(odom_x, odom_y, odom_theta)
            self.x = odom_x
            self.y = odom_y
            self.theta = odom_theta
            return

        # 记录原始轮式里程计（用于漂移监控）
        self._raw_wheel_x = odom_x
        self._raw_wheel_y = odom_y
        self._raw_wheel_theta = odom_theta

        self.last_wheel_x = odom_x
        self.last_wheel_y = odom_y
        self.last_wheel_theta = odom_theta

    def update(self, local_points: List[Tuple[float, float]],
               timestamp: float = 0.0,
               dx_wheel: float = 0.0,
               dy_wheel: float = 0.0,
               dtheta_wheel: float = 0.0) -> Tuple[bool, float, float, float]:
        """
        粒子滤波更新一帧

        Args:
            local_points: 当前帧的激光点云（局部坐标）
            timestamp: 时间戳
            dx_wheel: 轮式里程计 x 增量（mm），由外部计算传入
            dy_wheel: 轮式里程计 y 增量（mm）
            dtheta_wheel: 轮式里程计角度增量（rad）

        Returns:
            success, x, y, theta
        """
        if self.map_grid is None or len(self.particles) == 0:
            return False, self.x, self.y, self.theta

        if len(local_points) < 10:
            return True, self.x, self.y, self.theta

        self.frame_count += 1

        # 1. 运动传播（加噪声）—— 使用外部传入的真实增量，只执行一次
        self._motion_update(dx_wheel, dy_wheel, dtheta_wheel)

        # 3. 观测更新：根据点云匹配地图给粒子打分
        self._observation_update(local_points)

        # 4. 归一化权重
        self._normalize_weights()

        # 5. 计算有效粒子数，决定是否需要重采样
        neff = self._effective_sample_size()
        if neff < self.num_particles * self.resample_threshold:
            self._resample()
            self.resample_count += 1

        # 6. 输出：加权平均位姿
        self._estimate_pose()

        # 7. 静止检测 + 自适应噪声
        self._update_stationary_status()

        # 8. 漂移预警：协方差持续膨胀时发出警告
        if self.frame_count % 20 == 0 and not self.is_stationary:
            std_xy = math.hypot(self.pose_std[0], self.pose_std[1])
            if std_xy > 30 or math.degrees(self.pose_std[2]) > 5:
                print(f"[PF] ⚠ 位姿协方差膨胀 std_xy={std_xy:.1f}mm "
                      f"std_θ={math.degrees(self.pose_std[2]):.1f}° "
                      f"— 定位可能漂移中")

        if self.frame_count % 50 == 0:
            # 漂移监控：比较 PF 估计 vs 原始轮式里程计
            drift_msg = ""
            if hasattr(self, '_raw_wheel_x'):
                raw_dx = self.x - self._raw_wheel_x
                raw_dy = self.y - self._raw_wheel_y
                raw_dtheta = math.degrees(self._angle_diff(self.theta, self._raw_wheel_theta))
                drift_dist = math.hypot(raw_dx, raw_dy)
                if drift_dist > 50 or abs(raw_dtheta) > 3:
                    drift_msg = f" ⚠漂移:{drift_dist:.0f}mm/{raw_dtheta:.1f}°"
                else:
                    drift_msg = f" ✓漂移:{drift_dist:.0f}mm/{raw_dtheta:.1f}°"

            print(f"[PF] 帧:{self.frame_count} 位姿:({self.x:.0f},{self.y:.0f}) "
                  f"θ:{math.degrees(self.theta):.1f}° "
                  f"std:({self.pose_std[0]:.1f},{self.pose_std[1]:.1f},"
                  f"{math.degrees(self.pose_std[2]):.1f}°) "
                  f"Neff:{neff:.0f}/{self.num_particles} "
                  f"静止:{self.is_stationary}{drift_msg}")

        return True, self.x, self.y, self.theta

    def _motion_update(self, dx: float, dy: float, dtheta: float):
        """运动模型：所有粒子按轮式增量传播 + 高斯噪声"""
        # === 关键修改：增量极小时认为是静止噪声，直接忽略，从源头消除漂移 ===
        is_input_stationary = (
            abs(dx) < self.stationary_dx_threshold and
            abs(dy) < self.stationary_dy_threshold and
            abs(dtheta) < self.stationary_dtheta_threshold
        )

        if is_input_stationary:
            # 静止时不传播任何运动增量，也不加运动噪声
            # 粒子位置完全冻结，仅靠观测模型（激光匹配）维持定位
            self.is_stationary = True
            return
        else:
            # 增量超过阈值，明确标记为非静止
            self.is_stationary = False

        # 根据静止状态选择噪声大小
        if self.is_stationary:
            noise_xy = self.motion_noise_xy_static
            noise_theta = self.motion_noise_theta_static
        else:
            noise_xy = self.motion_noise_xy_dynamic
            noise_theta = self.motion_noise_theta_dynamic

        for p in self.particles:
            # ★ 关键修复：ESP32 发来的 odom.x/odom.y 是绝对世界坐标，
            # 增量 dx/dy 已经是世界坐标系下的位移，不需要再按粒子朝向旋转。
            # 之前的 "粒子坐标系→世界坐标" 旋转是错误的 — 每次转弯都会
            # 让所有粒子朝错误方向移动，导致观测模型无法纠正，快速漂移。
            p.x += dx + np.random.normal(0, noise_xy)
            p.y += dy + np.random.normal(0, noise_xy)
            p.theta += dtheta + np.random.normal(0, noise_theta)

            # 归一化角度
            while p.theta > math.pi:
                p.theta -= 2 * math.pi
            while p.theta < -math.pi:
                p.theta += 2 * math.pi

    def _observation_update(self, local_points: List[Tuple[float, float]]):
        """观测模型：点云与地图匹配打分"""
        pts = np.array(local_points, dtype=np.float32)
        grid = self.map_grid
        size = grid.size
        offset = grid.offset
        res = grid.resolution
        log_odds = grid.log_odds
        occ_thresh = grid.occ_thresh
        free_thresh = grid.free_thresh

        for p in self.particles:
            cos_t = math.cos(p.theta)
            sin_t = math.sin(p.theta)
            R = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
            world_pts = (R @ pts.T).T + np.array([p.x, p.y], dtype=np.float32)

            score = 0.0
            for wx, wy in world_pts:
                mx = int(-wy / res + offset)
                my = int(-wx / res + offset)

                if mx < 0 or mx >= size or my < 0 or my >= size:
                    score += self.out_penalty
                else:
                    val = log_odds[my, mx]
                    if val > occ_thresh:
                        score += self.hit_score
                    elif val < free_thresh:
                        score += self.miss_penalty
                    # 未知区域不加分不扣分

            # 权重 = exp(avg_score × scale)，按点数归一化防止溢出
            # avg_score ∈ [-1.0, 4.0]，scale=2.0 使 best:median ≈ 400:1
            n = len(local_points)
            if n > 0:
                avg_score = score / n
                p.weight = math.exp(avg_score * 2.0)
            else:
                p.weight = 1.0

    def _normalize_weights(self):
        """归一化粒子权重"""
        total = sum(p.weight for p in self.particles)
        if total > 0:
            for p in self.particles:
                p.weight /= total
        else:
            # 全部权重为0，均匀分布
            w = 1.0 / len(self.particles)
            for p in self.particles:
                p.weight = w

    def _effective_sample_size(self) -> float:
        """计算有效粒子数 N_eff = 1 / sum(w^2)"""
        sum_w2 = sum(p.weight ** 2 for p in self.particles)
        if sum_w2 > 0:
            return 1.0 / sum_w2
        return 0.0

    def _resample(self):
        """系统重采样（Systematic Resampling）"""
        n = len(self.particles)
        new_particles = []

        # 累积权重
        cumsum = [0.0]
        for p in self.particles:
            cumsum.append(cumsum[-1] + p.weight)

        # 系统采样
        step = 1.0 / n
        start = np.random.uniform(0, step)
        idx = 0

        for i in range(n):
            u = start + i * step
            while idx < n and cumsum[idx + 1] < u:
                idx += 1
            if idx < n:
                new_p = self.particles[idx].copy()
                # 重采样后加少量噪声防止粒子退化
                if self.is_stationary:
                    # 静止时几乎不加噪声，防止假漂移
                    new_p.x += np.random.normal(0, 0.3)
                    new_p.y += np.random.normal(0, 0.3)
                    new_p.theta += np.random.normal(0, 0.001)
                else:
                    # 降低重采样噪声（10.0→4.0mm, 0.03→0.01rad），减少累积漂移
                    new_p.x += np.random.normal(0, 4.0)
                    new_p.y += np.random.normal(0, 4.0)
                    new_p.theta += np.random.normal(0, 0.01)
                new_particles.append(new_p)
            else:
                new_particles.append(self.particles[-1].copy())

        self.particles = new_particles

    def _estimate_pose(self):
        """加权平均估计位姿"""
        x_sum = y_sum = 0.0
        sin_sum = cos_sum = 0.0
        total_w = 0.0

        for p in self.particles:
            w = p.weight
            x_sum += p.x * w
            y_sum += p.y * w
            sin_sum += math.sin(p.theta) * w
            cos_sum += math.cos(p.theta) * w
            total_w += w

        if total_w > 0:
            self.x = x_sum / total_w
            self.y = y_sum / total_w
            self.theta = math.atan2(sin_sum / total_w, cos_sum / total_w)

        # 计算标准差
        var_x = sum(p.weight * (p.x - self.x) ** 2 for p in self.particles)
        var_y = sum(p.weight * (p.y - self.y) ** 2 for p in self.particles)
        var_t = sum(p.weight * (self._angle_diff(p.theta, self.theta)) ** 2
                    for p in self.particles)
        self.pose_std = (math.sqrt(var_x), math.sqrt(var_y), math.sqrt(var_t))

    def _update_stationary_status(self):
        """更新静止状态：根据位姿标准差判断"""
        # 如果位置标准差很小，认为已经收敛到静止状态
        if self.pose_std[0] < 10.0 and self.pose_std[1] < 10.0:
            self.stationary_counter += 1
        else:
            self.stationary_counter = 0

        self.is_stationary = self.stationary_counter >= self.stationary_threshold

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        """计算角度差（归一化到 [-pi, pi]）"""
        diff = a - b
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return diff

    def reset(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.particles = []
        self.wheel_initialized = False
        self.frame_count = 0
        self.resample_count = 0
        self.stationary_counter = 0
        self.is_stationary = True

    def get_pose(self) -> Tuple[float, float, float]:
        return self.x, self.y, self.theta