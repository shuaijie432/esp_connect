"""
navigator.py - 两层避障：A*全局规划 + 局部动态窗口避障
（融合静态地图与实时点云，适配230mm机器人半径，底盘支持平移）
"""

import math
import heapq
import time
import numpy as np
from typing import List, Tuple, Optional
from collections import deque


class Navigator:
    def __init__(self, mapper):
        self.mapper = mapper
        self.state = "IDLE"
        self.path = []
        self.waypoints = []
        self.current_wp = 0

        self.lookahead_min = 200.0
        self.lookahead_max = 500.0
        self.wp_threshold = 60.0
        self.final_threshold = 40.0

        # 机器人物理尺寸（mm）
        self.robot_width_mm = 260.0           # 27cm 车身宽度
        self.robot_length_mm = 260.0          # 27cm 车身长度
        self.robot_radius_mm = self.robot_width_mm / 2.0   # 130mm（半宽）
        # 对角线半长 = sqrt(130²+130²) ≈ 184mm，A* 膨胀必须覆盖整个车身
        self.robot_diag_half_mm = math.hypot(self.robot_width_mm / 2, self.robot_length_mm / 2)
        self.safety_margin_mm = 100.0          # DWA 额外安全余量（80→100）
        self.total_inflation_mm = self.robot_radius_mm + self.safety_margin_mm  # 230mm

        # ★ A* 硬膨胀 = 22格 = 220mm（覆盖对角线半长184mm + 36mm余量）
        #    确保车身蓝色方框的四个角也不会进入膨胀区
        self.obstacle_margin = 22

        # ★ DWA 碰撞框参数
        self.channel_margin_mm = 100.0        # 通道余量（80→100）
        self.replan_threshold = 200.0

        self.replan_interval = 1.0
        self.last_replan_time = 0.0
        self.dynamic_obstacle_dist = 500.0
        self.dynamic_obstacle_angle = math.radians(45)

        self.safety_distance = 800.0
        self.obstacle_sector = math.radians(45)

        self.stuck_timer = 0.0
        self.last_pos = None
        self.stuck_recovery_until = 0.0

        self._inflated_grid = None
        self._grid_version = -1
        self._cost_grid = None          # 距离变换代价网格
        self._center_cost_grid = None   # 居中代价网格
        self._center_gain = 50.0        # 居中代价增益

        # 规划时冻结的地图
        self._planned_grid = None
        self._planned_version = -1
        self._plan_frozen = False

        # 速度参数
        self.MAX_VX = 150.0
        self.MAX_VY = 130.0
        self.MAX_VW = 0.4

        self.KP_V = 0.5
        self.KP_W = 0.9

        self._coord_checked = False

        self.target_theta = None
        self._align_settle_until = 0.
        self._alignment_ack_sent = False
        self._send_alignment_ack = False
        self._align_stable_count = 0

        # 障碍物永久融合参数
        self._min_cluster_size = 1
        self._permanent_after_sec = 1.5
        self._expire_after_sec = 5.0
        self._obstacle_cost_gain = 120.0
        self._obstacle_cost_decay = 8.0
        self._static_cost_gain = 150.0
        self._static_cost_decay = 10.0
        self._static_cost_grid = None
        self._obstacle_fusion_interval = 0.5
        self._stable_obstacles = {}
        self._obstacle_id_counter = 0

        # 局部代价地图
        self._local_costmap_size = 80
        self._local_costmap_resolution = 10
        self._local_costmap = np.zeros((self._local_costmap_size, self._local_costmap_size), dtype=np.float32)
        self._local_obstacles = []
        self._local_costmap_center = (0, 0)
        self._local_costmap_valid = False

        # 障碍物短期记忆
        self._local_obstacle_memory = {}
        self._local_obstacle_memory_ttl = 15

        # 动态窗口避障参数
        self._dw_enabled = True
        self._uniform_clearance = 240.0         # 统一安全距离（200→240）
        self._dw_safe_distance = 180.0          # 安全距离（150→180）
        self._dw_critical_distance = 140.0      # 临界距离（100→140）
        self._dw_lateral_gain = 2.0

        # 卡住检测
        self._stuck_check_start = 0.0
        self._stuck_check_duration = 5.0
        self._stuck_pos_history = deque(maxlen=50)
        self._stuck_dist_threshold = 200.0
        self._total_replan_count = 0
        self._max_replan = 999

        # 合速度速率限制器（防止突变）
        self._last_output_speed = 0.0           # 上一帧下发的合速度
        self.MIN_COMBINED_SPEED = 85.0          # 最低合速度 mm/s
        self.MAX_SPEED_DELTA = 35.0             # 每步最大加速量 (100ms步长 → 350mm/s²)
        self.MAX_SPEED_DELTA_BRAKE = 60.0       # 每步最大减速量 (允许更快刹车)

        # 起点保护
        self._startup_align_phase = 2
        self._startup_align_stable = 0
        self._nav_count = 0
        self.safety_boost = 0.0

        try:
            from scipy.ndimage import binary_dilation, distance_transform_edt
            self._has_scipy = True
        except ImportError:
            self._has_scipy = False
            print("[NAV] ⚠ scipy 未安装！膨胀和代价网格将使用慢速回退，A* 性能会下降")

        try:
            if self.mapper.static_map_mode:
                self._precompute_inflation(self.mapper.map)
                occ_count = np.count_nonzero(self._inflated_grid) if self._inflated_grid is not None else 0
                print(f"[NAV] 静态地图膨胀栅格已预计算: {occ_count} 个栅格")
            else:
                print("[NAV] 非静态地图模式，跳过初始膨胀栅格预计算")
        except Exception as e:
            print(f"[NAV] 初始膨胀栅格计算失败: {e}")

    # ============================================================
    # 冻结/解冻规划地图
    # ============================================================
    def _freeze_planning_map(self):
        if self._plan_frozen and self._planned_grid is not None:
            print("[NAV] 规划地图保持冻结（复用已有快照，防止漂移）")
            return self._get_planning_inflated_grid()
        grid = self.mapper.map
        self._planned_grid = grid.log_odds.copy()
        self._planned_version = time.time()
        self._plan_frozen = True
        occ_count = np.count_nonzero(self._planned_grid > grid.occ_thresh)
        print(f"[NAV] 规划地图已冻结，静态占用栅格: {occ_count}")
        return self._get_planning_inflated_grid()

    def _unfreeze_planning_map(self):
        self._plan_frozen = False
        self._planned_grid = None
        self._planned_inflated = None
        self._total_replan_count = 0
        print("[NAV] 规划地图已解冻")

    def _build_pointcloud_occ_mask(self, robot_x, robot_y):
        grid = self.mapper.map
        size = grid.size
        occ_mask = np.zeros((size, size), dtype=bool)

        _, world_pts = self.mapper.get_latest_points()
        if not world_pts:
            return occ_mask

        for wx, wy in world_pts:
            dist = math.hypot(wx - robot_x, wy - robot_y)
            if dist < 80 or dist > 8000:
                continue

            mx, my = grid.world_to_map(wx, wy)
            if 0 <= mx < size and 0 <= my < size:
                occ_mask[my, mx] = True
        return occ_mask

    def _get_planning_inflated_grid(self):
        grid = self.mapper.map
        robot_x = self.mapper.pose.x
        robot_y = self.mapper.pose.y

        if self._plan_frozen and self._planned_grid is not None:
            log_odds = self._planned_grid
        else:
            log_odds = grid.log_odds

        static_occ = log_odds > grid.occ_thresh
        pc_occ = self._build_pointcloud_occ_mask(robot_x, robot_y)
        combined_occ = static_occ | pc_occ

        for obs_id, obs_data in self._stable_obstacles.items():
            for mx, my in obs_data['cells']:
                if 0 <= mx < grid.size and 0 <= my < grid.size:
                    combined_occ[my, mx] = True

        rx, ry = grid.world_to_map(robot_x, robot_y)
        pre_clear = self.obstacle_margin
        for dx in range(-pre_clear, pre_clear + 1):
            for dy in range(-pre_clear, pre_clear + 1):
                nx, ny = rx + dx, ry + dy
                if 0 <= nx < grid.size and 0 <= ny < grid.size:
                    combined_occ[ny, nx] = False

        try:
            from scipy.ndimage import binary_dilation
            inflated = binary_dilation(combined_occ, iterations=self.obstacle_margin)
        except ImportError:
            inflated = combined_occ.copy()
            for _ in range(self.obstacle_margin):
                padded = np.pad(inflated, 1, mode='constant', constant_values=False)
                inflated = (
                    padded[0:-2, 1:-1] |
                    padded[2:, 1:-1] |
                    padded[1:-1, 0:-2] |
                    padded[1:-1, 2:]
                )

        post_clear = max(self.obstacle_margin // 3, 4)
        for dx in range(-post_clear, post_clear + 1):
            for dy in range(-post_clear, post_clear + 1):
                nx, ny = rx + dx, ry + dy
                if 0 <= nx < grid.size and 0 <= ny < grid.size:
                    inflated[ny, nx] = False

        try:
            from scipy.ndimage import distance_transform_edt
            self._cost_grid = distance_transform_edt(~combined_occ)
            self._static_cost_grid = distance_transform_edt(~static_occ)
            gy, gx = np.gradient(self._cost_grid)
            self._center_cost_grid = np.sqrt(gx**2 + gy**2)
        except ImportError:
            self._cost_grid = None
            self._static_cost_grid = None
            self._center_cost_grid = None

        return inflated

    def _get_cost_grid(self):
        return self._cost_grid

    def _precompute_inflation(self, grid):
        if self._inflated_grid is not None and self._grid_version == id(grid.log_odds):
            return
        static_occ = grid.log_odds > grid.occ_thresh
        try:
            from scipy.ndimage import binary_dilation
            self._inflated_grid = binary_dilation(static_occ, iterations=self.obstacle_margin)
        except ImportError:
            self._inflated_grid = static_occ.copy()
            for _ in range(self.obstacle_margin):
                padded = np.pad(self._inflated_grid, 1, mode='constant', constant_values=False)
                self._inflated_grid = (
                    padded[0:-2, 1:-1] |
                    padded[2:, 1:-1] |
                    padded[1:-1, 0:-2] |
                    padded[1:-1, 2:]
                )
        rx, ry = grid.world_to_map(self.mapper.pose.x, self.mapper.pose.y)
        if 0 <= rx < grid.size and 0 <= ry < grid.size:
            self._inflated_grid[ry, rx] = False
        self._grid_version = id(grid.log_odds)

    # ============================================================
    # 障碍物永久融合（保持不变）
    # ============================================================
    def permanently_add_obstacles(self, world_points, min_cluster_size=None):
        if min_cluster_size is None:
            min_cluster_size = self._min_cluster_size

        grid = self.mapper.map
        new_cells = set()
        robot_x = self.mapper.pose.x
        robot_y = self.mapper.pose.y

        for wx, wy in world_points:
            dist_to_robot = math.hypot(wx - robot_x, wy - robot_y)
            if dist_to_robot < 200:
                continue
            mx, my = grid.world_to_map(wx, wy)
            if not (0 <= mx < grid.size and 0 <= my < grid.size):
                continue
            if grid.log_odds[my, mx] > grid.occ_thresh + 2.0:
                continue
            new_cells.add((mx, my))

        if not new_cells:
            return {'added': 0, 'updated': 0, 'stable': 0, 'permanent': 0, 'small': 0, 'newly_permanent': 0}

        big_obstacles_clusters = []
        small_obstacles_clusters = []
        processed = set()

        for mx, my in list(new_cells):
            if (mx, my) in processed:
                continue
            cluster = []
            queue = deque([(mx, my)])
            cluster_set = set()

            while queue and len(cluster) < 500:
                cx, cy = queue.popleft()
                if (cx, cy) in cluster_set:
                    continue
                cluster_set.add((cx, cy))
                cluster.append((cx, cy))
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1),
                               (-1, -1), (1, 1), (-1, 1), (1, -1)]:
                    nx, ny = cx + dx, cy + dy
                    if (nx, ny) in new_cells and (nx, ny) not in cluster_set:
                        queue.append((nx, ny))

            for c in cluster:
                processed.add(c)

            cluster_size_mm = len(cluster) * grid.resolution
            centroid = self._calc_cluster_centroid(cluster, grid)

            cluster_info = {
                'cells': cluster,
                'size': len(cluster),
                'size_mm': cluster_size_mm,
                'centroid': centroid,
            }

            if len(cluster) >= min_cluster_size:
                big_obstacles_clusters.append(cluster_info)
            else:
                small_obstacles_clusters.append(cluster_info)

        current_time = time.time()
        small_count = len(small_obstacles_clusters)

        updated_count = 0
        stable_count = 0
        permanent_count = 0
        newly_permanent = 0

        grid = self.mapper.map
        expired_ids = []
        for obs_id, obs_data in self._stable_obstacles.items():
            if current_time - obs_data['last_update'] > self._expire_after_sec:
                expired_ids.append(obs_id)

        permanent_cleared = False
        for obs_id in expired_ids:
            obs_data = self._stable_obstacles[obs_id]
            if obs_data.get('is_permanent', False):
                for mx, my in obs_data['cells']:
                    if 0 <= mx < grid.size and 0 <= my < grid.size:
                        grid.log_odds[my, mx] = grid.log_free
                permanent_cleared = True
            del self._stable_obstacles[obs_id]

        MAX_TRACKED = 30
        if len(self._stable_obstacles) > MAX_TRACKED:
            sorted_obs = sorted(
                self._stable_obstacles.items(),
                key=lambda x: (x[1].get('is_permanent', False), x[1]['last_update'])
            )
            to_remove = len(self._stable_obstacles) - MAX_TRACKED
            for obs_id, _ in sorted_obs[:to_remove]:
                obs_data = self._stable_obstacles[obs_id]
                if obs_data.get('is_permanent', False):
                    for mx, my in obs_data['cells']:
                        if 0 <= mx < grid.size and 0 <= my < grid.size:
                            grid.log_odds[my, mx] = grid.log_free
                    permanent_cleared = True
                del self._stable_obstacles[obs_id]

        if permanent_cleared and self._plan_frozen:
            self._unfreeze_planning_map()
            self._freeze_planning_map()

        matched_stable_ids = set()

        for cluster_info in big_obstacles_clusters:
            new_cx, new_cy = cluster_info['centroid']
            matched = False

            best_match_id = None
            best_match_dist = float('inf')

            for obs_id, obs_data in self._stable_obstacles.items():
                if obs_id in matched_stable_ids:
                    continue
                if obs_data.get('is_permanent', False):
                    stable_cx, stable_cy = obs_data['fixed_position']
                else:
                    stable_cx, stable_cy = obs_data['centroid']
                dist = math.hypot(new_cx - stable_cx, new_cy - stable_cy)
                if dist < best_match_dist:
                    best_match_dist = dist
                    best_match_id = obs_id

            if best_match_id is not None and best_match_dist < 800.0:
                obs_data = self._stable_obstacles[best_match_id]

                if obs_data.get('is_permanent', False):
                    obs_data['last_update'] = current_time
                    matched_stable_ids.add(best_match_id)
                    permanent_count += 1
                    matched = True

                else:
                    obs_data['centroid'] = (new_cx, new_cy)
                    obs_data['cells'] = set(cluster_info['cells'])
                    obs_data['last_update'] = current_time
                    matched_stable_ids.add(best_match_id)
                    updated_count += 1
                    matched = True

                    elapsed = current_time - obs_data['first_seen']
                    if elapsed >= self._permanent_after_sec:
                        obs_data['is_permanent'] = True
                        obs_data['fixed_position'] = (new_cx, new_cy)
                        for mx, my in obs_data['cells']:
                            if 0 <= mx < grid.size and 0 <= my < grid.size:
                                grid.log_odds[my, mx] = grid.log_occ
                        stable_count += 1
                        newly_permanent += 1
                    else:
                        pass

            if not matched:
                self._obstacle_id_counter += 1
                new_id = self._obstacle_id_counter
                self._stable_obstacles[new_id] = {
                    'centroid': (new_cx, new_cy),
                    'cells': set(cluster_info['cells']),
                    'first_seen': current_time,
                    'last_update': current_time,
                    'is_permanent': False,
                    'fixed_position': None
                }
                matched_stable_ids.add(new_id)
                updated_count += 1

        all_permanent_cells = set()
        for obs_id, obs_data in self._stable_obstacles.items():
            if obs_data.get('is_permanent', False):
                all_permanent_cells.update(obs_data['cells'])

        need_replan = (updated_count > 0 or newly_permanent > 0) and self._plan_frozen

        return {
            'added': 0,
            'updated': updated_count,
            'stable': stable_count,
            'permanent': permanent_count + newly_permanent,
            'small': small_count,
            'newly_permanent': newly_permanent,
        }

    def _calc_cluster_centroid(self, cells, grid):
        if not cells:
            return (0, 0)
        mx = sum(c[0] for c in cells) / len(cells)
        my = sum(c[1] for c in cells) / len(cells)
        return grid.map_to_world(int(mx), int(my))

    def get_stable_obstacles(self):
        return list(self._stable_obstacles.values())

    def get_stable_obstacle_cells(self):
        cells = set()
        for obs_data in self._stable_obstacles.values():
            if obs_data.get('is_permanent', False):
                cells.update(obs_data['cells'])
        return list(cells)

    def get_temporary_obstacle_cells(self):
        cells = set()
        for obs_data in self._stable_obstacles.values():
            if not obs_data.get('is_permanent', False):
                cells.update(obs_data['cells'])
        return list(cells)

    def _clear_dynamic_obstacles(self):
        self._stable_obstacles.clear()
        self._inflated_grid = None
        self._cost_grid = None
        self._grid_version = -1
        self._unfreeze_planning_map()
        print("[NAV] 所有动态障碍物已清除，规划地图已解冻")

    # ============================================================
    # 局部代价地图
    # ============================================================
    def _update_local_costmap(self, robot_x, robot_y, robot_theta):
        grid = self.mapper.map
        size = self._local_costmap_size
        half = size // 2
        res = self._local_costmap_resolution

        self._local_costmap.fill(0.0)

        _tracked_for_dwa = []
        for obs_id, obs_data in self._stable_obstacles.items():
            is_perm = obs_data.get('is_permanent', False)
            cost = 100.0 if is_perm else 50.0
            for mx, my in obs_data['cells']:
                wx, wy = grid.map_to_world(mx, my)
                dx = wx - robot_x
                dy = wy - robot_y
                dist = math.hypot(dx, dy)
                local_x = dx * math.cos(robot_theta) + dy * math.sin(robot_theta)
                local_y = -dx * math.sin(robot_theta) + dy * math.cos(robot_theta)
                lx = int(local_x / res + half)
                ly = int(local_y / res + half)
                if 0 <= lx < size and 0 <= ly < size:
                    self._local_costmap[ly, lx] = max(self._local_costmap[ly, lx], cost)
                if dist <= 2000:
                    key = (round(wx / 50.0), round(wy / 50.0))
                    _tracked_for_dwa.append((local_x, local_y, dist, key))

        _, world_pts = self.mapper.get_latest_points()
        self._local_obstacles = []
        now = time.time()

        current_keys = set()
        for wx, wy in world_pts:
            dx = wx - robot_x
            dy = wy - robot_y
            dist = math.hypot(dx, dy)
            if dist > 1500 or dist < 20:
                continue
            key = (round(wx / 50.0), round(wy / 50.0))
            current_keys.add(key)
            self._local_obstacle_memory[key] = (wx, wy, now)

        expired = [k for k, v in self._local_obstacle_memory.items()
                   if now - v[2] > self._local_obstacle_memory_ttl]
        for k in expired:
            del self._local_obstacle_memory[k]

        for key, (wx, wy, t) in self._local_obstacle_memory.items():
            dx = wx - robot_x
            dy = wy - robot_y
            dist = math.hypot(dx, dy)
            if dist > 1500:
                continue

            local_x = dx * math.cos(robot_theta) + dy * math.sin(robot_theta)
            local_y = -dx * math.sin(robot_theta) + dy * math.cos(robot_theta)

            age = now - t
            if key in current_keys:
                confidence = 1.0
            else:
                confidence = max(0.3, 1.0 - age / self._local_obstacle_memory_ttl)

            self._local_obstacles.append((local_x, local_y, dist))

            lx = int(local_x / res + half)
            ly = int(local_y / res + half)
            if 0 <= lx < size and 0 <= ly < size:
                cost = min(100.0, 5000.0 / max(dist, 50)) * confidence
                self._local_costmap[ly, lx] = max(self._local_costmap[ly, lx], cost)

        tracked_added = set()
        for local_x, local_y, dist, key in _tracked_for_dwa:
            if key in current_keys or key in tracked_added:
                continue
            tracked_added.add(key)
            self._local_obstacles.append((local_x, local_y, dist))

        self._inflate_local_costmap()
        self._local_costmap_center = (robot_x, robot_y)
        self._local_costmap_valid = True

    def _inflate_local_costmap(self):
        size = self._local_costmap_size
        inflated = self._local_costmap.copy()
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                shifted = np.roll(self._local_costmap, (dy, dx), axis=(0, 1))
                inflated = np.maximum(inflated, shifted * 0.7)
        self._local_costmap = inflated

    def _get_local_cost(self, local_x, local_y):
        if not self._local_costmap_valid:
            return 0.0
        half = self._local_costmap_size // 2
        res = self._local_costmap_resolution
        lx = int(local_x / res + half)
        ly = int(local_y / res + half)
        if 0 <= lx < self._local_costmap_size and 0 <= ly < self._local_costmap_size:
            return self._local_costmap[ly, lx]
        return 100.0

    # ============================================================
    # 标准 DWA
    # ============================================================
    def _dynamic_window_avoidance(self, x, y, theta, base_vx, base_vy, base_vw, emergency=False) -> Tuple[float, float, float]:
        self._update_local_costmap(x, y, theta)

        cos0, sin0 = math.cos(theta), math.sin(theta)
        obs_world = [(x + ox * cos0 - oy * sin0,
                       y + ox * sin0 + oy * cos0)
                      for ox, oy, _ in self._local_obstacles]

        PHYS_MARGIN = 0.0
        phys_hl = self.robot_length_mm / 2.0 + PHYS_MARGIN
        phys_hw = self.robot_width_mm / 2.0 + PHYS_MARGIN

        COMFORT_MARGIN = 220.0     # DWA 舒适框余量（175→220，匹配 A* 膨胀）
        comfort_hl = phys_hl + COMFORT_MARGIN
        comfort_hw = phys_hw + COMFORT_MARGIN

        dt = 0.3
        vx_min = max(-self.MAX_VX, base_vx - 430.0 * dt)
        vx_max = min(self.MAX_VX, base_vx + 430.0 * dt)
        vy_min = max(-self.MAX_VY, base_vy - 430.0 * dt)
        vy_max = min(self.MAX_VY, base_vy + 430.0 * dt)
        vw_min = max(-self.MAX_VW, base_vw - 2.0 * dt)
        vw_max = min(self.MAX_VW, base_vw + 2.0 * dt)

        predict_time = 1.5
        time_step = 0.15

        if self.waypoints and self.current_wp < len(self.waypoints):
            target = self._find_lookahead_point(x, y, self._adaptive_lookahead())
        elif self.waypoints:
            target = self.waypoints[-1]
        else:
            target = None

        if target:
            dx_w = target[0] - x
            dy_w = target[1] - y
            dx_local = dx_w * cos0 + dy_w * sin0
            dy_local = -dx_w * sin0 + dy_w * cos0

            ideal_vx = np.clip(self.KP_V * dx_local, vx_min, vx_max)
            ideal_vy = np.clip(self.KP_V * dy_local * 0.7, vy_min, vy_max)
            target_angle = math.atan2(dy_local, max(dx_local, 1.0))
            ideal_vw = np.clip(self.KP_W * target_angle, vw_min, vw_max)

            near_vx = np.linspace(
                max(vx_min, ideal_vx - (ideal_vx - vx_min) * 0.4),
                min(vx_max, ideal_vx + (vx_max - ideal_vx) * 0.4), 5)
            vx_samples = np.unique(np.concatenate([[vx_min], near_vx, [vx_max]]))

            near_vy = np.linspace(
                max(vy_min, ideal_vy - (ideal_vy - vy_min) * 0.4),
                min(vy_max, ideal_vy + (vy_max - ideal_vy) * 0.4), 3)
            vy_samples = np.unique(np.concatenate([[vy_min], near_vy, [vy_max]]))

            near_vw = np.linspace(
                max(vw_min, ideal_vw - 0.15),
                min(vw_max, ideal_vw + 0.15), 4)
            vw_samples = np.unique(np.concatenate([[vw_min], near_vw, [vw_max]]))
        else:
            vx_samples = np.linspace(vx_min, vx_max, 7)
            vy_samples = np.linspace(vy_min, vy_max, 5)
            vw_samples = np.linspace(vw_min, vw_max, 5)

        comfort_safe = []
        tight_ok = []
        all_cands = []

        for cvx in vx_samples:
            for cvy in vy_samples:
                for cvw in vw_samples:
                    px, py, ptheta = x, y, theta
                    hard_collision = False
                    comfort_collision = False
                    min_cl = float('inf')
                    steps = int(predict_time / time_step)

                    ct0, st0 = cos0, sin0
                    for ob_wx, ob_wy in obs_world:
                        dx_l0 = (ob_wx - x) * ct0 + (ob_wy - y) * st0
                        dy_l0 = -(ob_wx - x) * st0 + (ob_wy - y) * ct0
                        d0 = max(abs(dx_l0) - comfort_hl, abs(dy_l0) - comfort_hw)
                        min_cl = min(min_cl, d0)
                        if abs(dx_l0) < phys_hl and abs(dy_l0) < phys_hw:
                            hard_collision = True
                            break
                        if abs(dx_l0) < comfort_hl and abs(dy_l0) < comfort_hw:
                            comfort_collision = True
                    if hard_collision:
                        cand = {'vx': cvx, 'vy': cvy, 'vw': cvw,
                                'hard_collision': True, 'comfort_collision': True,
                                'clearance': min_cl, 'path_dist': 99999.0}
                        all_cands.append(cand)
                        continue

                    for _ in range(steps):
                        ptheta += cvw * time_step
                        px += cvx * math.cos(ptheta) * time_step - cvy * math.sin(ptheta) * time_step
                        py += cvx * math.sin(ptheta) * time_step + cvy * math.cos(ptheta) * time_step

                        ct, st = math.cos(ptheta), math.sin(ptheta)
                        for ob_wx, ob_wy in obs_world:
                            dx_l = (ob_wx - px) * ct + (ob_wy - py) * st
                            dy_l = -(ob_wx - px) * st + (ob_wy - py) * ct

                            d_comfort = max(abs(dx_l) - comfort_hl, abs(dy_l) - comfort_hw)
                            min_cl = min(min_cl, d_comfort)

                            if abs(dx_l) < phys_hl and abs(dy_l) < phys_hw:
                                hard_collision = True
                                break
                            if abs(dx_l) < comfort_hl and abs(dy_l) < comfort_hw:
                                comfort_collision = True
                        if hard_collision:
                            break

                    path_dist = math.hypot(target[0] - px, target[1] - py) if target else 0.0
                    path_dev = self._calc_path_deviation(px, py) if self.waypoints and len(self.waypoints) >= 2 else 0.0

                    cand = {'vx': cvx, 'vy': cvy, 'vw': cvw,
                            'hard_collision': hard_collision,
                            'comfort_collision': comfort_collision,
                            'clearance': min_cl, 'path_dist': path_dist,
                            'path_dev': path_dev}
                    all_cands.append(cand)

                    if not hard_collision:
                        tight_ok.append(cand)
                        if not comfort_collision:
                            comfort_safe.append(cand)

        if comfort_safe:
            cl_vals = [c['clearance'] for c in comfort_safe]
            pd_vals = [c['path_dist'] for c in comfort_safe]
            dev_vals = [c['path_dev'] for c in comfort_safe]
            cl_min_v, cl_max_v = min(cl_vals), max(cl_vals)
            pd_min_v, pd_max_v = min(pd_vals), max(pd_vals)
            dev_min_v, dev_max_v = min(dev_vals), max(dev_vals)
            cl_r = cl_max_v - cl_min_v + 1e-6
            pd_r = pd_max_v - pd_min_v + 1e-6
            dev_r = dev_max_v - dev_min_v + 1e-6
            for c in comfort_safe:
                cl_norm = (c['clearance'] - cl_min_v) / cl_r
                pd_norm = 1.0 - (c['path_dist'] - pd_min_v) / pd_r
                dev_norm = 1.0 - (c['path_dev'] - dev_min_v) / dev_r
                c['score'] = pd_norm * 0.50 + dev_norm * 0.25 + cl_norm * 0.25
            best = max(comfort_safe, key=lambda c: c['score'])
        elif tight_ok:
            cl_vals = [c['clearance'] for c in tight_ok]
            pd_vals = [c['path_dist'] for c in tight_ok]
            dev_vals = [c['path_dev'] for c in tight_ok]
            vy_abs_vals = [abs(c['vy']) for c in tight_ok]
            cl_min, cl_max = min(cl_vals), max(cl_vals)
            pd_min, pd_max = min(pd_vals), max(pd_vals)
            dev_min, dev_max = min(dev_vals), max(dev_vals)
            vy_max = max(vy_abs_vals) + 1e-6
            cl_r = cl_max - cl_min + 1e-6
            pd_r = pd_max - pd_min + 1e-6
            dev_r = dev_max - dev_min + 1e-6
            for c in tight_ok:
                c['score'] = (c['clearance'] - cl_min) / cl_r * 0.35 \
                           + (1.0 - (c['path_dist'] - pd_min) / pd_r) * 0.25 \
                           + (1.0 - (c['path_dev'] - dev_min) / dev_r) * 0.25 \
                           + abs(c['vy']) / vy_max * 0.15
            best = max(tight_ok, key=lambda c: c['score'])
        elif all_cands:
            best = max(all_cands, key=lambda c: c['clearance'])
        else:
            return base_vx, base_vy, base_vw

        vx, vy, vw = best['vx'], best['vy'], best['vw']

        if best.get('comfort_collision', False):
            cl = max(-COMFORT_MARGIN, min(0.0, best['clearance']))
            speed_scale = 0.5 + 0.5 * (cl + COMFORT_MARGIN) / COMFORT_MARGIN
            vx *= speed_scale
            vy *= speed_scale

        vx = max(-self.MAX_VX, min(self.MAX_VX, vx))
        vy = max(-self.MAX_VY, min(self.MAX_VY, vy))
        vw = max(-self.MAX_VW, min(self.MAX_VW, vw))

        if not hasattr(self, '_last_dwa_vx'):
            self._last_dwa_vx = base_vx
            self._last_dwa_vy = base_vy
            self._last_dwa_vw = base_vw
        alpha = 0.50
        vx_smooth = alpha * vx + (1 - alpha) * self._last_dwa_vx
        vy_smooth = alpha * vy + (1 - alpha) * self._last_dwa_vy
        vw_smooth = alpha * vw + (1 - alpha) * self._last_dwa_vw

        max_dv = 80.0
        max_dw = 0.25
        dvx = vx_smooth - self._last_dwa_vx
        dvy = vy_smooth - self._last_dwa_vy
        dvw = vw_smooth - self._last_dwa_vw
        vx = self._last_dwa_vx + max(-max_dv, min(max_dv, dvx))
        vy = self._last_dwa_vy + max(-max_dv, min(max_dv, dvy))
        vw = self._last_dwa_vw + max(-max_dw, min(max_dw, dvw))

        self._last_dwa_vx = vx
        self._last_dwa_vy = vy
        self._last_dwa_vw = vw

        return vx, vy, vw


    # ============================================================
    # 导航主循环
    # ============================================================
    def update(self, x: float, y: float, theta: float) -> Tuple[float, float, float]:
        if self.state in ("IDLE", "FAILED", "DONE"):
            return 0.0, 0.0, 0.0

        if self._startup_align_phase < 2:
            return self._startup_alignment_step(x, y, theta)

        if self.state == "ALIGNING":
            return self._align_heading_step(x, y, theta)

        if self._reached_goal(x, y):
            self._nav_count += 1
            if self.target_theta is not None:
                self.state = "ALIGNING"
                print(f"[NAV] 到达目标位置，开始对准角度 {math.degrees(self.target_theta):.1f}°")
                return 0.0, 0.0, 0.0
            else:
                self.state = "DONE"
                self.final_approach_dist = 0.0
                self.safety_boost = 0.0
                self._nav_start_time = 0.0
                print("[NAV] 到达目标！")
                return 0.0, 0.0, 0.0

        now = time.time()

        self._stuck_pos_history.append((x, y, now))

        if self._stuck_check_start == 0.0:
            self._stuck_check_start = now

        if len(self._stuck_pos_history) >= 10 and now - self._stuck_check_start > self._stuck_check_duration:
            old_x, old_y, old_t = self._stuck_pos_history[0]
            if now - old_t > self._stuck_check_duration:
                moved = math.hypot(x - old_x, y - old_y)
                if moved < self._stuck_dist_threshold and self._total_replan_count < self._max_replan:
                    print(f"[NAV] 卡住检测！{self._stuck_check_duration:.0f}秒内只移动了 {moved:.0f}mm，触发重规划 (第{self._total_replan_count+1}次)")
                    if self.waypoints:
                        target = self.waypoints[-1]
                        if self.set_target(target[0], target[1], self.target_theta):
                            self._total_replan_count += 1
                            return 0.0, 0.0, 0.0
                        else:
                            self.state = "FAILED"
                            return 0.0, 0.0, 0.0
                while self._stuck_pos_history and now - self._stuck_pos_history[0][2] > self._stuck_check_duration:
                    self._stuck_pos_history.popleft()

        if self.state not in ("ALIGNING", "EVADE_EMERGENCY", "AVOIDING") and self.last_pos:
            moved = math.hypot(x - self.last_pos[0], y - self.last_pos[1])
            if moved > 30:
                self.stuck_timer = now
            elif now - self.stuck_timer > 2.0 and now > self.stuck_recovery_until:
                print("[NAV] 瞬时卡住检测触发！触发重规划")
                self.stuck_recovery_until = now + 2.0
                self.stuck_timer = now
                if self.waypoints:
                    target = self.waypoints[-1]
                    self.set_target(target[0], target[1], self.target_theta)
                return 0.0, 0.0, 0.0
        self.last_pos = (x, y)

        self._sync_waypoint_index(x, y)

        lateral_error = self._calc_lateral_error(x, y, theta)
        if lateral_error > 400.0 and now - self.last_replan_time > 5.0:
            if self.waypoints:
                if self.set_target(self.waypoints[-1][0], self.waypoints[-1][1], self.target_theta):
                    self.last_replan_time = now
                    return 0.0, 0.0, 0.0
        elif lateral_error > 300.0:
            print(f"[NAV] 横向偏差 {lateral_error:.0f}mm 较大，依赖回归修正")


        obstacle_level = self._check_obstacle_level(x, y, theta)
        front_dist = self._get_front_distance(x, y, theta)
        left_dist = self._get_left_distance(x, y, theta)
        right_dist = self._get_right_distance(x, y, theta)
        rear_dist = self._get_rear_distance(x, y, theta)

        if front_dist < 400.0 or obstacle_level != "CLEAR":
            self.stuck_timer = now

        self.state = "FOLLOWING"
        base_vx, base_vy, base_vw = self._pure_pursuit_step(x, y, theta)

        # ---- 前方减速 ----
        if front_dist < 350.0:
            speed_scale = max(0.15, (front_dist - 120.0) / 230.0)
            base_vx = base_vx * speed_scale
            base_vy = base_vy * speed_scale
            if front_dist < 220.0:
                print(f"[BRAKE] 前方障碍物 {front_dist:.0f}mm，速度缩放至 {speed_scale:.2f}")

        # ---- 侧面减速 ----
        side_min = min(left_dist, right_dist)
        if side_min < 250.0:
            side_scale = max(0.2, (side_min - 80.0) / 170.0)
            base_vy = base_vy * side_scale
            if side_min < 150.0:
                print(f"[SIDE_BRAKE] 侧面障碍物 {side_min:.0f}mm，横向速度缩放至 {side_scale:.2f}")

        # ---- 后方防撞 ----
        if rear_dist < 200.0 and base_vx < 0:
            rear_scale = max(0.1, (rear_dist - 50.0) / 150.0)
            base_vx = base_vx * rear_scale
            print(f"[REAR_BRAKE] 后方障碍物 {rear_dist:.0f}mm，后退速度缩放至 {rear_scale:.2f}")

        dist_to_goal = math.hypot(self.waypoints[-1][0] - x, self.waypoints[-1][1] - y) \
                       if self.waypoints else float('inf')

        if self._dw_enabled and dist_to_goal >= 200.0:
            if obstacle_level == "EMERGENCY":
                vx, vy, vw = self._evade_obstacle(x, y, theta)
                self.state = "EVADE_EMERGENCY"
            else:
                vx, vy, vw = self._dynamic_window_avoidance(
                    x, y, theta, base_vx, base_vy, base_vw)
                if obstacle_level == "CAUTION":
                    self.state = "AVOIDING"
        else:
            vx, vy, vw = base_vx, base_vy, base_vw

        # ============================================================
        # 合速度平滑限幅：保证 ≥85mm/s 且不会突变
        # ============================================================
        speed = math.hypot(vx, vy)

        if speed > 0.01:
            # 1. 强制最低合速度（紧急避障后退时跳过，直接发原始低速）
            if speed < self.MIN_COMBINED_SPEED and self.state != "EVADE_EMERGENCY":
                scale = self.MIN_COMBINED_SPEED / speed
                vx *= scale
                vy *= scale
                speed = self.MIN_COMBINED_SPEED

            # 2. 速率限制器：防止相邻两帧合速度突变
            delta = speed - self._last_output_speed

            # 加速上限
            max_delta = self.MAX_SPEED_DELTA
            # 减速上限（允许更快刹车）
            max_brake = self.MAX_SPEED_DELTA_BRAKE

            if delta > max_delta:
                speed = self._last_output_speed + max_delta
            elif delta < -max_brake:
                speed = self._last_output_speed - max_brake

            if speed > 0.01:
                old_speed = math.hypot(vx, vy)
                if old_speed > 0.01:
                    scale = speed / old_speed
                    vx *= scale
                    vy *= scale

            self._last_output_speed = speed
        else:
            # 输出为零（停止），允许立刻降为零
            self._last_output_speed = 0.0

        return vx, vy, vw

    def _calc_lateral_error(self, x, y, theta) -> float:
        if not self.waypoints or self.current_wp >= len(self.waypoints):
            return 0.0
        target = self.waypoints[self.current_wp]
        dx = target[0] - x
        dy = target[1] - y
        path_angle = math.atan2(dy, dx)
        lateral = abs(math.sin(path_angle - theta) * math.hypot(dx, dy))
        return lateral

    def check_path_blocked(self) -> bool:
        if not self.waypoints or self.current_wp >= len(self.waypoints):
            return False
        grid = self.mapper.map
        static_occ = grid.log_odds > grid.occ_thresh
        _, world_pts = self.mapper.get_latest_points()
        pc_occ = set()
        if world_pts:
            for wx, wy in world_pts:
                mx, my = grid.world_to_map(wx, wy)
                if 0 <= mx < grid.size and 0 <= my < grid.size:
                    pc_occ.add((mx, my))

        tracked_occ = set()
        for obs_data in self._stable_obstacles.values():
            for mx, my in obs_data['cells']:
                if 0 <= mx < grid.size and 0 <= my < grid.size:
                    tracked_occ.add((mx, my))

        margin = int(round(self.channel_margin_mm / grid.resolution))
        for i in range(self.current_wp, min(self.current_wp + 3, len(self.waypoints))):
            wx, wy = self.waypoints[i]
            mx, my = grid.world_to_map(wx, wy)
            for dx in range(-margin, margin + 1):
                for dy in range(-margin, margin + 1):
                    nx, ny = mx + dx, my + dy
                    if 0 <= nx < grid.size and 0 <= ny < grid.size:
                        if static_occ[ny, nx] or (nx, ny) in pc_occ or (nx, ny) in tracked_occ:
                            return True
        return False

    def _get_front_distance(self, x, y, theta) -> float:
        _, world_pts = self.mapper.get_latest_points()
        if not world_pts:
            return float('inf')
        front_min = float('inf')
        for wx, wy in world_pts:
            dx = wx - x
            dy = wy - y
            dist = math.hypot(dx, dy)
            if dist > 2000 or dist < 20:
                continue
            angle = self._normalize_angle(math.atan2(dy, dx) - theta)
            if abs(angle) < math.radians(70):
                front_min = min(front_min, dist)
        return front_min

    def _get_left_distance(self, x, y, theta) -> float:
        """左侧障碍物最近距离（70° ~ 110°）"""
        _, world_pts = self.mapper.get_latest_points()
        if not world_pts:
            return float('inf')
        left_min = float('inf')
        for wx, wy in world_pts:
            dx = wx - x
            dy = wy - y
            dist = math.hypot(dx, dy)
            if dist > 2000 or dist < 20:
                continue
            angle = self._normalize_angle(math.atan2(dy, dx) - theta)
            if math.radians(70) <= angle <= math.radians(110):
                left_min = min(left_min, dist)
        return left_min

    def _get_right_distance(self, x, y, theta) -> float:
        """右侧障碍物最近距离（-110° ~ -70°）"""
        _, world_pts = self.mapper.get_latest_points()
        if not world_pts:
            return float('inf')
        right_min = float('inf')
        for wx, wy in world_pts:
            dx = wx - x
            dy = wy - y
            dist = math.hypot(dx, dy)
            if dist > 2000 or dist < 20:
                continue
            angle = self._normalize_angle(math.atan2(dy, dx) - theta)
            if math.radians(-110) <= angle <= math.radians(-70):
                right_min = min(right_min, dist)
        return right_min

    def _get_rear_distance(self, x, y, theta) -> float:
        """后方障碍物最近距离（|angle| > 110°）"""
        _, world_pts = self.mapper.get_latest_points()
        if not world_pts:
            return float('inf')
        rear_min = float('inf')
        for wx, wy in world_pts:
            dx = wx - x
            dy = wy - y
            dist = math.hypot(dx, dy)
            if dist > 2000 or dist < 20:
                continue
            angle = self._normalize_angle(math.atan2(dy, dx) - theta)
            if abs(angle) > math.radians(110):
                rear_min = min(rear_min, dist)
        return rear_min

    def set_target(self, x: float, y: float, theta: float = None) -> bool:
        self.target_theta = theta

        start = (self.mapper.pose.x, self.mapper.pose.y)
        goal = (x, y)

        self._freeze_planning_map()

        raw_path = self._astar(start, goal)
        if not raw_path:
            self.state = "FAILED"
            self._unfreeze_planning_map()
            return False

        self.path = raw_path
        inflated = self._get_planning_inflated_grid()
        self.waypoints = self._smooth_path(raw_path, inflated)
        self.current_wp = 0

        self.state = "FOLLOWING"
        self.last_replan_time = time.time()
        self.stuck_timer = time.time()
        self.last_pos = (self.mapper.pose.x, self.mapper.pose.y)

        self._nav_start_time = time.time()

        self._stuck_check_start = time.time()
        self._stuck_pos_history.clear()

        self._align_settle_until = 0.0
        self._alignment_ack_sent = False
        self._send_alignment_ack = False
        self._align_stable_count = 0

        self._startup_align_phase = 2
        self._startup_align_stable = 0

        # 重置速度状态，确保新导航从零开始平滑加速
        self._last_output_speed = 0.0
        if hasattr(self, '_last_dwa_vx'):
            del self._last_dwa_vx
            del self._last_dwa_vy
            del self._last_dwa_vw

        print(f"[NAV] 目标已设置: ({x:.0f}, {y:.0f}), 路径点: {len(self.waypoints)}个, 规划已冻结")
        return True

    def cancel(self):
        self.state = "IDLE"
        self.path = []
        self.waypoints = []
        self.current_wp = 0
        self.target_theta = None
        self._align_settle_until = 0.0
        self._alignment_ack_sent = False
        self._send_alignment_ack = False
        self._align_stable_count = 0
        self._startup_align_phase = 2
        self._startup_align_stable = 0
        self._nav_count = 0
        self.final_approach_dist = 0.0
        self.safety_boost = 0.0
        self._nav_start_time = 0.0
        # 重置速度状态，防止下次导航继承旧速度
        self._last_output_speed = 0.0
        if hasattr(self, '_last_dwa_vx'):
            del self._last_dwa_vx
            del self._last_dwa_vy
            del self._last_dwa_vw
        self._unfreeze_planning_map()
        print("[NAV] 导航已取消，规划地图已解冻")

    # ============================================================
    # 起点保护
    # ============================================================
    def _startup_alignment_step(self, x: float, y: float, theta: float) -> Tuple[float, float, float]:
        if self._nav_count == 0:
            if not hasattr(self, '_startup_start_pos'):
                self._startup_start_pos = (x, y)

            frame = self._startup_align_stable
            self._startup_align_stable += 1

            moved = math.hypot(x - self._startup_start_pos[0],
                               y - self._startup_start_pos[1])

            if frame >= 3 and moved > 150.0:
                self._startup_align_phase = 2
                delattr(self, '_startup_start_pos')
                print(f"[STARTUP] 起点保护完成 (移动 {moved:.0f}mm) → 开始路径跟踪")
                return 0.0, 0.0, 0.0
            return 100.0, -100.0, 0.0

        if not self.waypoints or len(self.waypoints) < 2:
            self._startup_align_phase = 2
            return 0.0, 0.0, 0.0

        wx0, wy0 = self.waypoints[0]
        wx1, wy1 = self.waypoints[1]
        path_dir = math.atan2(wy1 - wy0, wx1 - wx0)

        angle_diff = self._normalize_angle(path_dir - theta)
        abs_diff = abs(angle_diff)

        if abs_diff < math.radians(15.0):
            self._startup_align_stable += 1
            if self._startup_align_stable >= 3:
                self._startup_align_phase = 2
                print(f"[STARTUP] 朝向对齐完成 (偏差 {math.degrees(abs_diff):.1f}°) → 开始路径跟踪")
                return 0.0, 0.0, 0.0
        else:
            self._startup_align_stable = 0

        vw = max(-self.MAX_VW, min(self.MAX_VW, self.KP_W * angle_diff))
        return 0.0, 0.0, vw

    def _align_heading_step(self, x: float, y: float, theta: float) -> Tuple[float, float, float]:
        angle_diff = self._normalize_angle(self.target_theta - theta)
        abs_diff = abs(angle_diff)
        now = time.time()

        if abs_diff >= math.radians(8.0):
            self._align_settle_until = 0.0
            self._alignment_ack_sent = False
            self._align_stable_count = 0

            # 固定角速度 8°/s，方向由偏差符号决定
            vw = math.radians(8.0)
            if angle_diff < 0:
                vw = -vw
            return 0.0, 0.0, vw

        if self._align_stable_count < 3:
            self._align_stable_count += 1
            return 0.0, 0.0, 0.0

        if self._align_settle_until == 0.0:
            self._align_settle_until = now + 0.5
            print(f"[NAV] 角度已稳定，等待500ms后发送完成标志... 当前: {math.degrees(theta):.1f}°")
            return 0.0, 0.0, 0.0

        if now < self._align_settle_until:
            return 0.0, 0.0, 0.0

        if not self._alignment_ack_sent:
            self._alignment_ack_sent = True
            self._send_alignment_ack = True
            print(f"[NAV] 500ms 等待结束，准备发送导航完成标志位...")
            return 0.0, 0.0, 0.0

        return 0.0, 0.0, 0.0

    def _get_path_direction(self, x: float, y: float, theta: float) -> float:
        if not self.waypoints or self.current_wp >= len(self.waypoints):
            return None

        wx, wy = self.waypoints[self.current_wp]
        if math.hypot(wx - x, wy - y) < self.wp_threshold * 2:
            if self.current_wp + 1 < len(self.waypoints):
                wx, wy = self.waypoints[self.current_wp + 1]

        dx = wx - x
        dy = wy - y
        path_angle = math.atan2(dy, dx)
        return self._normalize_angle(path_angle - theta)

    def _calc_path_deviation(self, x: float, y: float) -> float:
        if not self.waypoints or len(self.waypoints) < 2:
            return 0.0

        min_dist = float('inf')
        start_idx = max(0, self.current_wp - 1)
        end_idx = min(len(self.waypoints) - 1, self.current_wp + 2)

        for i in range(start_idx, end_idx):
            x1, y1 = self.waypoints[i]
            x2, y2 = self.waypoints[i + 1]
            dx = x2 - x1
            dy = y2 - y1
            seg_len = math.hypot(dx, dy)
            if seg_len < 1:
                dist = math.hypot(x - x1, y - y1)
            else:
                t = max(0, min(1, ((x - x1) * dx + (y - y1) * dy) / (seg_len * seg_len)))
                proj_x = x1 + t * dx
                proj_y = y1 + t * dy
                dist = math.hypot(x - proj_x, y - proj_y)
            min_dist = min(min_dist, dist)

        return min_dist

    def _get_path_direction_at(self, x: float, y: float):
        if not self.waypoints or len(self.waypoints) < 2:
            return None

        min_dist = float('inf')
        best_dir = None

        start_idx = max(0, self.current_wp - 1)
        end_idx = min(len(self.waypoints) - 1, self.current_wp + 3)

        for i in range(start_idx, end_idx):
            x1, y1 = self.waypoints[i]
            x2, y2 = self.waypoints[i + 1]
            dx = x2 - x1
            dy = y2 - y1
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq < 1:
                continue

            t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / seg_len_sq))
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy
            dist = (x - proj_x) ** 2 + (y - proj_y) ** 2

            if dist < min_dist:
                min_dist = dist
                best_dir = math.atan2(dy, dx)

        return best_dir

    def _calc_path_projection(self, x, y):
        if not self.waypoints or len(self.waypoints) < 2:
            return 0.0, (x, y)

        min_dist = float('inf')
        best_proj = (x, y)
        best_seg_vec = (0, 0)

        start_idx = max(0, self.current_wp - 1)
        end_idx = min(len(self.waypoints) - 1, self.current_wp + 3)

        for i in range(start_idx, end_idx):
            x1, y1 = self.waypoints[i]
            x2, y2 = self.waypoints[i + 1]

            dx = x2 - x1
            dy = y2 - y1
            seg_len_sq = dx * dx + dy * dy

            if seg_len_sq < 1:
                proj_x, proj_y = x1, y1
            else:
                t = max(0, min(1, ((x - x1) * dx + (y - y1) * dy) / seg_len_sq))
                proj_x = x1 + t * dx
                proj_y = y1 + t * dy

            dist = math.hypot(x - proj_x, y - proj_y)
            if dist < min_dist:
                min_dist = dist
                best_proj = (proj_x, proj_y)
                best_seg_vec = (dx, dy)

        vec_robot = (x - best_proj[0], y - best_proj[1])
        if math.hypot(best_seg_vec[0], best_seg_vec[1]) < 1e-6:
            signed_lateral = 0.0
        else:
            cross = best_seg_vec[0] * vec_robot[1] - best_seg_vec[1] * vec_robot[0]
            signed_lateral = cross / math.hypot(best_seg_vec[0], best_seg_vec[1])
        return signed_lateral, best_proj

    def _sync_waypoint_index(self, x: float, y: float):
        if self.current_wp >= len(self.waypoints) - 1:
            return

        wx, wy = self.waypoints[self.current_wp]
        dist = math.hypot(wx - x, wy - y)

        if dist > 400:
            best_idx = self.current_wp
            best_dist = dist
            for i in range(self.current_wp + 1, len(self.waypoints)):
                nx, ny = self.waypoints[i]
                nd = math.hypot(nx - x, ny - y)
                if nd < best_dist:
                    best_dist = nd
                    best_idx = i
                if nd > best_dist * 2.0:
                    break
            if best_idx > self.current_wp:
                skipped = best_idx - self.current_wp
                self.current_wp = best_idx
                print(f"[NAV] DWA绕行后跳过{skipped}个路径点 -> [{self.current_wp}]")
                return

        if dist > 300 and self.current_wp + 1 < len(self.waypoints):
            nx, ny = self.waypoints[self.current_wp + 1]
            nd = math.hypot(nx - x, ny - y)
            if nd < dist:
                self.current_wp += 1
                print(f"[NAV] 推进到 [{self.current_wp}]")

        if self.current_wp + 1 < len(self.waypoints):
            cx, cy = self.waypoints[self.current_wp]
            nx, ny = self.waypoints[self.current_wp + 1]
            v1x, v1y = nx - cx, ny - cy
            v2x, v2y = x - cx, y - cy
            dot = v1x * v2x + v1y * v2y
            if dot > 0 and math.hypot(v2x, v2y) > self.wp_threshold:
                self.current_wp += 1
                print(f"[NAV] 越过路径点 [{self.current_wp}]")

    # ============================================================
    # Pure Pursuit + 路径回归修正
    # ============================================================
    def _pure_pursuit_step(self, x, y, theta) -> Tuple[float, float, float]:
        while self.current_wp < len(self.waypoints) - 1:
            wx, wy = self.waypoints[self.current_wp]
            if math.hypot(wx - x, wy - y) < self.wp_threshold:
                self.current_wp += 1
                print(f"[NAV] 到达路径点 [{self.current_wp}]")
            else:
                break

        if self.current_wp >= len(self.waypoints):
            return 0.0, 0.0, 0.0

        lookahead = self._adaptive_lookahead()
        target = self._find_lookahead_point(x, y, lookahead)
        if target is None:
            target = self.waypoints[-1]

        dx = target[0] - x
        dy = target[1] - y
        dist = math.hypot(dx, dy)

        is_final = self.current_wp >= len(self.waypoints) - 1
        if is_final and dist < self.final_threshold:
            return 0.0, 0.0, 0.0

        target_angle = math.atan2(dy, dx)
        angle_diff = self._normalize_angle(target_angle - theta)
        abs_angle = abs(angle_diff)

        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        local_x = dx * cos_t + dy * sin_t
        local_y = -dx * sin_t + dy * cos_t

        max_vx = self.MAX_VX
        max_vy = self.MAX_VY
        max_vw = self.MAX_VW

        if is_final and dist < 400:
            scale = max(0.15, dist / 800.0)
            max_vx *= scale
            max_vy *= scale
            max_vw *= scale

        kp_v = self.KP_V
        if is_final and dist < 400:
            kp_v = 0.25

        _, world_pts = self.mapper.get_latest_points()
        channel_width = float('inf')
        _channel_lateral_shift = 0.0
        if world_pts:
            half_len = self.robot_length_mm * 0.5
            full_len = self.robot_length_mm

            probe_offsets = [0.0, half_len, full_len]
            probe_left = [float('inf'), float('inf'), float('inf')]
            probe_right = [float('inf'), float('inf'), float('inf')]

            for wx, wy in world_pts:
                for pi, offset in enumerate(probe_offsets):
                    if offset == 0.0:
                        px, py = x, y
                    else:
                        px = x + offset * math.cos(theta)
                        py = y + offset * math.sin(theta)
                    pdx = wx - px
                    pdy = wy - py
                    pd = math.hypot(pdx, pdy)
                    if pd > 1500 or pd < 50:
                        continue
                    angle = self._normalize_angle(math.atan2(pdy, pdx) - theta)
                    a_deg = math.degrees(angle)
                    if 10 <= a_deg < 90:
                        probe_left[pi] = min(probe_left[pi], pd)
                    elif -90 < a_deg <= -10:
                        probe_right[pi] = min(probe_right[pi], pd)

            left_wall   = probe_left[0]
            right_wall  = probe_right[0]
            mid_left    = probe_left[1]
            mid_right   = probe_right[1]
            front_left  = probe_left[2]
            front_right = probe_right[2]

            if left_wall < float('inf') and right_wall < float('inf'):
                channel_width = left_wall + right_wall

                if channel_width < 550:
                    channel_scale = max(0.50, channel_width / 550.0)
                    max_vx *= channel_scale
                    max_vy *= channel_scale

                    asymmetry = left_wall - right_wall
                    _channel_lateral_shift = np.clip(asymmetry * 0.7, -max_vy * 0.50, max_vy * 0.50)

                    fwd_valid = (mid_left < float('inf') and mid_right < float('inf') and
                                 front_left < float('inf') and front_right < float('inf'))
                    min_fwd_passage = float('inf')
                    if fwd_valid:
                        mid_passage = mid_left + mid_right
                        front_passage = front_left + front_right
                        min_fwd_passage = min(mid_passage, front_passage)

                        if min_fwd_passage == front_passage:
                            fwd_center_offset = (front_right - front_left) / 2.0
                        else:
                            fwd_center_offset = (mid_right - mid_left) / 2.0

                        center_correction = np.clip(fwd_center_offset * 0.6, -max_vy * 0.40, max_vy * 0.40)
                        _channel_lateral_shift += center_correction
                        _channel_lateral_shift = np.clip(_channel_lateral_shift, -max_vy * 0.60, max_vy * 0.60)

                        min_safe_width = self.robot_width_mm + self.safety_margin_mm * 1.3
                        if min_fwd_passage < min_safe_width:
                            tight_scale = max(0.18, min_fwd_passage / min_safe_width)
                            max_vx *= tight_scale
                            max_vy *= tight_scale
                            if min_fwd_passage < 420:
                                print(f"[CHANNEL] 前方极窄{min_fwd_passage:.0f}mm < 安全{min_safe_width:.0f}mm "
                                      f"中心偏移{fwd_center_offset:.0f}mm 大幅降速至{tight_scale:.2f}")

                    if channel_width < 500:
                        fwd_str = f" 前方最窄{min_fwd_passage:.0f}mm" if fwd_valid else ""
                        print(f"[CHANNEL] 窄道{channel_width:.0f}mm L={left_wall:.0f} R={right_wall:.0f}"
                              f"{fwd_str} 降速{channel_scale:.2f} 侧移{_channel_lateral_shift:.0f}")

            elif left_wall < float('inf') and left_wall < 450:
                right_open = (right_wall == float('inf') or right_wall > 700)
                if right_open:
                    shift_mag = min(70.0, max(30.0, (450.0 - left_wall) * 0.55))
                    _channel_lateral_shift = -shift_mag
                    print(f"[SIDE] 左侧贴墙{left_wall:.0f}mm 右侧开阔 → 大幅右移 vy={_channel_lateral_shift:.0f}")
                else:
                    _channel_lateral_shift = -min(40.0, (450.0 - left_wall) * 0.35)
            elif right_wall < float('inf') and right_wall < 450:
                left_open = (left_wall == float('inf') or left_wall > 700)
                if left_open:
                    shift_mag = min(70.0, max(30.0, (450.0 - right_wall) * 0.55))
                    _channel_lateral_shift = shift_mag
                    print(f"[SIDE] 右侧贴墙{right_wall:.0f}mm 左侧开阔 → 大幅左移 vy={_channel_lateral_shift:.0f}")
                else:
                    _channel_lateral_shift = min(40.0, (450.0 - right_wall) * 0.35)

        vx_raw = kp_v * local_x

        vy_raw = self.KP_V * local_y * 0.7

        v_mag = math.hypot(vx_raw, vy_raw)
        if v_mag > max_vx:
            ratio = max_vx / v_mag
            vx_raw *= ratio
            vy_raw *= ratio

        vx = max(-max_vx, min(max_vx, vx_raw))
        vy = max(-max_vy, min(max_vy, vy_raw))

        if _channel_lateral_shift != 0.0:
            vy += _channel_lateral_shift
            v_mag = math.hypot(vx, vy)
            if v_mag > max_vx:
                ratio = max_vx / v_mag
                vx *= ratio
                vy *= ratio

        path_lateral_err, path_proj = self._calc_path_projection(x, y)
        regress_gain = 0.5 if channel_width < 550 else 0.7
        if abs(path_lateral_err) > 10.0 and len(self.waypoints) > 2:
            reg_dx = path_proj[0] - x
            reg_dy = path_proj[1] - y
            reg_dist = math.hypot(reg_dx, reg_dy)
            if reg_dist > 5:
                vy_correction = -path_lateral_err * regress_gain
                max_vy_correction = self.MAX_VY * 0.15 if channel_width < 550 else self.MAX_VY * 0.25
                vy_correction = max(-max_vy_correction, min(max_vy_correction, vy_correction))
                vy += vy_correction

                reg_local_x = reg_dx * cos_t + reg_dy * sin_t
                if reg_local_x > 0:
                    vx += min(20.0, reg_local_x * 0.2)

                v_mag = math.hypot(vx, vy)
                if v_mag > max_vx:
                    ratio = max_vx / v_mag
                    vx *= ratio
                    vy *= ratio

        if self.current_wp < len(self.waypoints) - 1:
            next_wp = self.waypoints[self.current_wp + 1]
            path_dx = next_wp[0] - self.waypoints[self.current_wp][0]
            path_dy = next_wp[1] - self.waypoints[self.current_wp][1]
            path_angle = math.atan2(path_dy, path_dx)
        else:
            path_angle = target_angle

        path_angle_diff = self._normalize_angle(path_angle - theta)
        vw_err = angle_diff * 0.9 + path_angle_diff * 0.1
        vw = max(-max_vw, min(max_vw, self.KP_W * vw_err))

        if abs_angle > math.radians(60):
            vx *= 0.6
            if abs_angle > math.radians(90):
                print(f"[TURN] 角度差 {math.degrees(abs_angle):.1f}°，目标在后方，减速")

        return vx, vy, vw

    def _check_obstacle_level(self, x, y, theta) -> str:
        _, world_pts = self.mapper.get_latest_points()
        if not world_pts:
            return "CLEAR"

        front_min = float('inf')
        left_min = float('inf')
        right_min = float('inf')
        rear_min = float('inf')

        for wx, wy in world_pts:
            dx = wx - x
            dy = wy - y
            dist = math.hypot(dx, dy)
            if dist > 1500:
                continue
            angle = self._normalize_angle(math.atan2(dy, dx) - theta)

            if abs(angle) < math.radians(45):
                front_min = min(front_min, dist)
            elif math.radians(45) <= angle < math.radians(135):
                left_min = min(left_min, dist)
            elif math.radians(-135) < angle <= math.radians(-45):
                right_min = min(right_min, dist)
            else:
                rear_min = min(rear_min, dist)

        if front_min < 180 or left_min < 180 or right_min < 180:
            return "EMERGENCY"
        if front_min < 320 or left_min < 320 or right_min < 320 or rear_min < 320:
            return "CAUTION"

        return "CLEAR"

    def _evade_obstacle(self, x, y, theta) -> Tuple[float, float, float]:
        _, world_pts = self.mapper.get_latest_points()
        if not world_pts:
            return 0.0, 0.0, 0.0

        path_dir = self._get_path_direction(x, y, theta)
        if path_dir is None:
            path_dir = 0.0

        front_min = float('inf')
        fleft_min = float('inf')
        fright_min = float('inf')
        left_min = float('inf')
        right_min = float('inf')

        for wx, wy in world_pts:
            dx = wx - x
            dy = wy - y
            dist = math.hypot(dx, dy)
            if dist > 2000:
                continue

            angle = self._normalize_angle(math.atan2(dy, dx) - theta)
            angle_deg = math.degrees(angle)

            if abs(angle_deg) < 20:
                front_min = min(front_min, dist)
            elif 20 <= angle_deg < 60:
                fleft_min = min(fleft_min, dist)
            elif -60 < angle_deg <= -20:
                fright_min = min(fright_min, dist)
            elif 60 <= angle_deg < 110:
                left_min = min(left_min, dist)
            elif -110 < angle_deg <= -60:
                right_min = min(right_min, dist)

        if front_min > 450 and fleft_min > 400 and fright_min > 400:
            if left_min < 280:
                side_speed = min(80.0, max(30.0, (280.0 - left_min) * 0.5))
                print(f"[EVADE] 左侧障碍物 {left_min:.0f}mm，向右避让 vy={side_speed:.0f}")
                return 60.0, -side_speed, 0.0
            if right_min < 280:
                side_speed = min(80.0, max(30.0, (280.0 - right_min) * 0.5))
                print(f"[EVADE] 右侧障碍物 {right_min:.0f}mm，向左避让 vy={side_speed:.0f}")
                return 60.0, side_speed, 0.0
            base_vx, base_vy, base_vw = self._pure_pursuit_step(x, y, theta)
            return base_vx * 0.3, base_vy * 0.3, base_vw * 0.3


        best_dir = None
        best_score = -float('inf')

        test_dirs = []
        if front_min < 280:
            for priority_offset in [90, -90, 75, -75, 60, -60, 45, -45, 30, -30, 15, -15, 0, 120, -120]:
                test_dirs.append(self._normalize_angle(path_dir + math.radians(priority_offset)))
        elif right_min < 280:
            for priority_offset in [90, 75, 60, 45, 30, 15, 0, -15, -30, -45, -60, -75, -90, 120, -120]:
                test_dirs.append(self._normalize_angle(path_dir + math.radians(priority_offset)))
        elif left_min < 280:
            for priority_offset in [-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90, 120, -120]:
                test_dirs.append(self._normalize_angle(path_dir + math.radians(priority_offset)))
        else:
            for priority_offset in [0, 15, -15, 30, -30, 45, -45, 60, -60, 75, -75, 90, -90, 120, -120]:
                test_dirs.append(self._normalize_angle(path_dir + math.radians(priority_offset)))

        check_dists = [100, 200, 350]
        half_diag_phys = math.hypot(self.robot_length_mm / 2.0, self.robot_width_mm / 2.0)
        safe_radius = half_diag_phys + self.safety_margin_mm + self.safety_boost

        for test_rad in test_dirs:
            safe = True
            min_obstacle_dist = float('inf')

            for cd in check_dists:
                check_x = x + cd * math.cos(theta + test_rad)
                check_y = y + cd * math.sin(theta + test_rad)
                for wx, wy in world_pts:
                    d = math.hypot(wx - check_x, wy - check_y)
                    if d < safe_radius:
                        safe = False
                        break
                    min_obstacle_dist = min(min_obstacle_dist, d)
                if not safe:
                    break

            if not safe:
                continue

            dir_diff = abs(self._normalize_angle(test_rad - path_dir))
            score = 200 - math.degrees(dir_diff) * 1.2 + min_obstacle_dist * 0.05

            if front_min < 400:
                side_component = abs(math.sin(test_rad))
                score += side_component * 80
            elif front_min < 550:
                if abs(math.sin(test_rad)) > 0.5:
                    score += 30
            if right_min < 450:
                if math.sin(test_rad) > 0:
                    score += (450.0 - right_min) * 0.3
            if left_min < 450:
                if math.sin(test_rad) < 0:
                    score += (450.0 - left_min) * 0.3
            if right_min < 600 and left_min < 600:
                asymmetry = right_min - left_min
                score += math.sin(test_rad) * asymmetry * 0.2

            if score > best_score:
                best_score = score
                best_dir = test_rad

        if best_dir is not None:
            if front_min < 400:
                speed = 80.0
            elif front_min < 600:
                speed = 150.0
            else:
                speed = 120.0

            vx = speed * math.cos(best_dir)
            vy = speed * math.sin(best_dir)
            vw = 0.0
            print(f"[EVADE] 选择方向={math.degrees(best_dir):.0f}° vx={vx:.0f} vy={vy:.0f}")
            return vx, vy, vw

        print("[EVADE] 所有方向受阻，慢速后退")
        return -50.0, 0.0, 0.0


    def get_status(self) -> str:
        if self.state == "IDLE":
            return "导航: 空闲"
        elif self.state == "DONE":
            return "导航: 到达目标 ✓"
        elif self.state == "FAILED":
            return "导航: 失败 ✗"
        elif self.state == "STUCK_RECOVERY":
            return "导航: 卡住恢复 ↩"
        elif self.state == "EVADE_EMERGENCY":
            return "导航: 紧急避障 🚨"
        elif self.state == "AVOIDING":
            return f"导航: 绕行 [{self.current_wp}/{len(self.waypoints)}]"
        elif self.state == "ALIGNING":
            remain = math.degrees(self._normalize_angle(self.target_theta - self.mapper.pose.theta))
            return f"导航: 对准角度 [{remain:+.1f}°]"
        else:
            frozen = " [冻结]" if self._plan_frozen else ""
            return f"导航: 跟踪 [{self.current_wp}/{len(self.waypoints)}]{frozen}"

    def is_active(self) -> bool:
        return self.state not in ("IDLE", "DONE", "FAILED")

    def get_waypoints(self) -> List[Tuple[float, float]]:
        return list(self.waypoints)

    def get_inflated_grid(self):
        grid = self.mapper.map
        inflated = self._get_planning_inflated_grid()
        return inflated, grid.size, grid.resolution, grid.offset

    def get_raw_path(self) -> List[Tuple[float, float]]:
        return list(self.path)

    def get_lookahead_point(self) -> Optional[Tuple[float, float]]:
        if not self.waypoints or self.current_wp >= len(self.waypoints):
            return None
        lookahead = self._adaptive_lookahead()
        return self._find_lookahead_point(
            self.mapper.pose.x, self.mapper.pose.y, lookahead
        )

    def get_obstacle_level(self) -> str:
        if self.state in ("IDLE", "DONE", "FAILED"):
            return "CLEAR"
        return self._check_obstacle_level(
            self.mapper.pose.x, self.mapper.pose.y, self.mapper.pose.theta
        )

    def get_obstacle_sectors(self) -> dict:
        _, world_pts = self.mapper.get_latest_points()
        if not world_pts:
            return {}

        x, y, theta = self.mapper.pose.x, self.mapper.pose.y, self.mapper.pose.theta
        sectors = {
            'front': float('inf'),
            'f_left': float('inf'),
            'f_right': float('inf'),
            'left': float('inf'),
            'right': float('inf'),
        }
        for wx, wy in world_pts:
            dx = wx - x
            dy = wy - y
            dist = math.hypot(dx, dy)
            if dist > 1500:
                continue
            angle = self._normalize_angle(math.atan2(dy, dx) - theta)
            a_deg = math.degrees(angle)

            if abs(a_deg) < 25:
                sectors['front'] = min(sectors['front'], dist)
            elif 25 <= a_deg < 70:
                sectors['f_left'] = min(sectors['f_left'], dist)
            elif -70 < a_deg <= -25:
                sectors['f_right'] = min(sectors['f_right'], dist)
            elif 70 <= a_deg < 110:
                sectors['left'] = min(sectors['left'], dist)
            elif -110 < a_deg <= -70:
                sectors['right'] = min(sectors['right'], dist)

        return sectors

    # ============================================================
    # A* 和膨胀函数
    # ============================================================
    def _astar(self, start, goal) -> List[Tuple[float, float]]:
        grid = self.mapper.map
        size = grid.size

        occ_count = np.count_nonzero(grid.log_odds > grid.occ_thresh)
        free_count = np.count_nonzero(grid.log_odds < grid.free_thresh)
        print(f"[A*] 地图状态: 占用={occ_count}, 空闲={free_count}, 未知={size*size-occ_count-free_count}")

        sx, sy = grid.world_to_map(start[0], start[1])
        gx, gy = grid.world_to_map(goal[0], goal[1])
        print(f"[A*] 起点世界: ({start[0]:.0f}, {start[1]:.0f}) -> 地图: ({sx}, {sy})")
        print(f"[A*] 终点世界: ({goal[0]:.0f}, {goal[1]:.0f}) -> 地图: ({gx}, {gy})")

        inflated = self._get_planning_inflated_grid()
        cost_grid = self._get_cost_grid()
        if inflated is None:
            print("[A*] 错误：无法获取膨胀栅格")
            return []

        def in_bounds(mx, my):
            return 0 <= mx < size and 0 <= my < size

        if not in_bounds(sx, sy):
            print(f"[A*] 起点越界: ({sx}, {sy})")
            return []

        if inflated[sy, sx]:
            print(f"[A*] 警告：起点栅格({sx},{sy})被占用，强制清空")
            inflated[sy, sx] = False

        if not in_bounds(gx, gy):
            print(f"[A*] 终点越界: ({gx}, {gy})，搜索最近可用点...")
            free_gx, free_gy = self._find_nearest_free_inflated(
                max(0, min(size - 1, gx)),
                max(0, min(size - 1, gy)),
                size,
                inflated
            )
            if free_gx is None:
                print("[A*] 未找到可用终点")
                return []
            gx, gy = free_gx, free_gy
            print(f"[A*] 修正终点到: ({gx}, {gy})")
        elif inflated[gy, gx]:
            print(f"[A*] 终点栅格({gx},{gy})被占用，搜索最近空闲点...")
            free_gx, free_gy = self._find_nearest_free_inflated(gx, gy, size, inflated)
            if free_gx is None:
                print("[A*] 未找到可用终点")
                return []
            gx, gy = free_gx, free_gy
            print(f"[A*] 修正终点到: ({gx}, {gy})")

        g_score = np.full((size, size), np.inf)
        f_score = np.full((size, size), np.inf)
        visited = np.zeros((size, size), dtype=bool)
        came_from = {}

        g_score[sy, sx] = 0.0
        f_score[sy, sx] = math.hypot(gx - sx, gy - sy)
        open_set = [(f_score[sy, sx], sx, sy)]

        neighbors = [
            (0, 1, 1.0), (1, 0, 1.0), (0, -1, 1.0), (-1, 0, 1.0),
            (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)
        ]

        nodes_expanded = 0
        while open_set:
            _, cx, cy = heapq.heappop(open_set)
            if visited[cy, cx]:
                continue
            visited[cy, cx] = True
            nodes_expanded += 1

            if (cx, cy) == (gx, gy):
                path = []
                cur = (cx, cy)
                while cur in came_from:
                    path.append(grid.map_to_world(cur[0], cur[1]))
                    cur = came_from[cur]
                path.append(start)
                path.reverse()
                path[-1] = goal
                print(f"[A*] 找到路径！节点数: {len(path)}, 扩展节点: {nodes_expanded}")
                return path

            for dx, dy, cost in neighbors:
                nx, ny = cx + dx, cy + dy
                if not in_bounds(nx, ny) or visited[ny, nx]:
                    continue
                if inflated[ny, nx]:
                    continue

                if cost_grid is not None:
                    dist_cells = cost_grid[ny, nx]
                    occ_penalty = self._obstacle_cost_gain / (1.0 + dist_cells / self._obstacle_cost_decay)
                else:
                    occ_penalty = 0.0

                if self._static_cost_grid is not None:
                    static_dist = self._static_cost_grid[ny, nx]
                    static_penalty = self._static_cost_gain / (1.0 + static_dist / self._static_cost_decay)
                else:
                    static_penalty = 0.0

                # ==============================================================
                # === 修改点 1：修复 A* 居中代价（原梯度幅值失效，改用距离倒数惩罚） ===
                # ==============================================================
                if cost_grid is not None:
                    dist_cells = cost_grid[ny, nx]
                    # 距离越远，代价越小；距离越近（0），代价急剧增大，强制路径走在中间
                    center_penalty = self._center_gain / (dist_cells + 1.0)
                else:
                    center_penalty = 0.0
                # ==============================================================

                tentative = g_score[cy, cx] + cost + occ_penalty + static_penalty + center_penalty
                if tentative < g_score[ny, nx]:
                    came_from[(nx, ny)] = (cx, cy)
                    g_score[ny, nx] = tentative
                    f = tentative + math.hypot(gx - nx, gy - ny)
                    f_score[ny, nx] = f
                    heapq.heappush(open_set, (f, nx, ny))

        print(f"[A*] 未找到路径！扩展节点: {nodes_expanded}")
        return []

    def _find_nearest_free_inflated(self, gx, gy, size, inflated_grid):
        q = deque([(gx, gy)])
        visited = np.zeros((size, size), dtype=bool)
        visited[gy, gx] = True

        while q:
            cx, cy = q.popleft()
            if 0 <= cx < size and 0 <= cy < size and not inflated_grid[cy, cx]:
                return cx, cy
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < size and 0 <= ny < size and not visited[ny, nx]:
                    visited[ny, nx] = True
                    q.append((nx, ny))
        return None, None

    def _smooth_path(self, path: List[Tuple[float, float]], inflated_grid=None) -> List[Tuple[float, float]]:
        if len(path) <= 2:
            return path

        if inflated_grid is None:
            inflated_grid = self._get_planning_inflated_grid()
        grid = self.mapper.map

        sampled = [path[0]]
        for p in path[1:-1]:
            if math.hypot(p[0] - sampled[-1][0], p[1] - sampled[-1][1]) > 80:
                sampled.append(p)
        sampled.append(path[-1])

        smooth = [list(p) for p in sampled]

        # ======================================================================
        # === 修改点 2：削弱平滑的内切效应。降低迭代次数、降低拉向中心权重，===
        # ===          防止拐角处“圆弧化”而紧贴墙壁。                      ===
        # ======================================================================
        for _ in range(5):  # 原为 10 次，容易过度内切
            for i in range(1, len(smooth) - 1):
                ax = (smooth[i-1][0] + smooth[i+1][0]) * 0.5
                ay = (smooth[i-1][1] + smooth[i+1][1]) * 0.5
                smooth[i][0] = 0.95 * smooth[i][0] + 0.05 * ax  # 原 0.8 / 0.2，极易向墙拉
                smooth[i][1] = 0.95 * smooth[i][1] + 0.05 * ay
        # ======================================================================

        restored = []
        cost_grid = self._cost_grid
        for i, p in enumerate(smooth):
            mx, my = grid.world_to_map(p[0], p[1])
            if 0 <= mx < grid.size and 0 <= my < grid.size:
                if inflated_grid[my, mx]:
                    pushed_x, pushed_y = p[0], p[1]
                    pushed = False
                    for _ in range(20):
                        pmx, pmy = grid.world_to_map(pushed_x, pushed_y)
                        if not (0 <= pmx < grid.size and 0 <= pmy < grid.size):
                            break
                        if not inflated_grid[pmy, pmx]:
                            pushed = True
                            break
                        best_dx, best_dy = 0, 0
                        best_cost = -1.0
                        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)]:
                            nx, ny = pmx + dx, pmy + dy
                            if 0 <= nx < grid.size and 0 <= ny < grid.size:
                                c = cost_grid[ny, nx] if cost_grid is not None else 0
                                if c > best_cost:
                                    best_cost = c
                                    best_dx, best_dy = dx, dy
                        wx_step = best_dx * grid.resolution
                        wy_step = best_dy * grid.resolution
                        if best_dx == 0 and best_dy == 0:
                            break
                        pushed_x += wx_step
                        pushed_y += wy_step
                    if pushed:
                        restored.append((pushed_x, pushed_y))
                        print(f"[SMOOTH] 路径点 {i} 平滑后侵入膨胀区，已沿梯度推开至安全区")
                    else:
                        orig = sampled[i]
                        restored.append((orig[0], orig[1]))
                        print(f"[SMOOTH] 路径点 {i} 无法推开，回退到原始采样点")
                    continue
            restored.append((p[0], p[1]))

        # ======================================================================
        # === 修改点 3：调大软排斥阈值。原 21格=210mm 余量太少，                   ===
        # ===          改 35格=350mm，强制路径远离障碍物，留足转弯空间。          ===
        # ======================================================================
        if cost_grid is not None:
            SOFT_MARGIN = self.obstacle_margin * 2 + 5  # 35格=350mm
            REPEL_GAIN = 0.5
            repelled = []
            for p in restored:
                mx, my = grid.world_to_map(p[0], p[1])
                if 0 <= mx < grid.size and 0 <= my < grid.size:
                    d = cost_grid[my, mx]
                    if d < SOFT_MARGIN:
                        gx = 0.0
                        gy = 0.0
                        if 0 < mx < grid.size - 1:
                            gx = cost_grid[my, mx+1] - cost_grid[my, mx-1]
                        if 0 < my < grid.size - 1:
                            gy = cost_grid[my+1, mx] - cost_grid[my-1, mx]
                        gmag = math.hypot(gx, gy)
                        if gmag > 0.01:
                            gx /= gmag
                            gy /= gmag
                        push_mm = (SOFT_MARGIN - d) * REPEL_GAIN * grid.resolution
                        repelled.append((p[0] + gx * push_mm, p[1] + gy * push_mm))
                        continue
                repelled.append(p)
            restored = repelled
        # ======================================================================

        DENSE_SPACING = 50.0
        EXTRA_INSERTS = 2

        densified = [restored[0]]
        skipped_inflated = 0
        for i in range(1, len(restored)):
            prev = restored[i - 1]
            curr = restored[i]
            seg_dist = math.hypot(curr[0] - prev[0], curr[1] - prev[1])
            total_inserts = max(0, int(seg_dist / DENSE_SPACING) - 1) + EXTRA_INSERTS
            for j in range(1, total_inserts + 1):
                t = j / (total_inserts + 1)
                ix = prev[0] + t * (curr[0] - prev[0])
                iy = prev[1] + t * (curr[1] - prev[1])
                imx, imy = grid.world_to_map(ix, iy)
                if 0 <= imx < grid.size and 0 <= imy < grid.size:
                    if inflated_grid[imy, imx]:
                        skipped_inflated += 1
                        continue
                densified.append((ix, iy))
            densified.append(curr)

        if skipped_inflated > 0:
            print(f"[SMOOTH] 加密时跳过 {skipped_inflated} 个侵入膨胀区的插值点")
        print(f"[SMOOTH] 路径点: {len(path)} → {len(sampled)}(采样) → "
              f"{len(restored)}(平滑) → {len(densified)}(加密@{DENSE_SPACING:.0f}mm+{EXTRA_INSERTS})")
        return densified

    def _adaptive_lookahead(self) -> float:
        if self.current_wp >= len(self.waypoints) - 1:
            return self.lookahead_min

        p0 = (self.mapper.pose.x, self.mapper.pose.y)
        p1 = self.waypoints[self.current_wp]
        p2 = self.waypoints[min(self.current_wp + 1, len(self.waypoints) - 1)]

        v1 = (p1[0] - p0[0], p1[1] - p0[1])
        v2 = (p2[0] - p1[0], p2[1] - p1[1])
        d1 = math.hypot(*v1)
        d2 = math.hypot(*v2)
        if d1 < 1 or d2 < 1:
            return self.lookahead_min

        cos_a = max(-1.0, min(1.0, (v1[0]*v2[0] + v1[1]*v2[1]) / (d1*d2)))
        turn_angle = math.acos(cos_a)
        ratio = 1.0 - (turn_angle / math.pi)
        return self.lookahead_min + ratio * (self.lookahead_max - self.lookahead_min)

    def _find_lookahead_point(self, x, y, lookahead):
        n = len(self.waypoints)
        if n == 0:
            return None

        for i in range(self.current_wp, n):
            dist = math.hypot(self.waypoints[i][0] - x, self.waypoints[i][1] - y)
            if dist >= lookahead:
                if i == 0:
                    return self.waypoints[0]
                p1 = self.waypoints[i-1]
                p2 = self.waypoints[i]
                d1 = math.hypot(p1[0]-x, p1[1]-y)
                d2 = dist
                if d2 - d1 < 1e-6:
                    return p2
                t = (lookahead - d1) / (d2 - d1)
                t = max(0.0, min(1.0, t))
                return (p1[0] + t*(p2[0]-p1[0]), p1[1] + t*(p2[1]-p1[1]))

        return self.waypoints[-1] if n else None

    def _obstacle_ahead(self, x, y, theta) -> bool:
        local_pts, world_pts = self.mapper.get_latest_points()
        if not world_pts:
            return False

        for wx, wy in world_pts:
            dx = wx - x
            dy = wy - y
            dist = math.hypot(dx, dy)
            if dist > self.safety_distance or dist < 80:
                continue
            angle = math.atan2(dy, dx) - theta
            angle = self._normalize_angle(angle)
            if abs(angle) < self.obstacle_sector:
                return True
        return False

    def _path_blocked_by_obstacle(self, x, y, theta) -> bool:
        if not self.waypoints or self.current_wp >= len(self.waypoints):
            return False

        ahead_points = []
        for i in range(self.current_wp, min(self.current_wp + 5, len(self.waypoints))):
            wx, wy = self.waypoints[i]
            dist = math.hypot(wx - x, wy - y)
            if dist < self.dynamic_obstacle_dist:
                ahead_points.append((wx, wy, dist))

        if not ahead_points:
            return False

        _, world_pts = self.mapper.get_latest_points()
        if not world_pts:
            return False

        for ox, oy in world_pts:
            for px, py, _ in ahead_points:
                dist_to_path = math.hypot(ox - px, oy - py)
                if dist_to_path < 250:
                    dx = ox - x
                    dy = oy - y
                    dist = math.hypot(dx, dy)
                    if dist < self.dynamic_obstacle_dist and dist > 100:
                        angle = math.atan2(dy, dx) - theta
                        angle = self._normalize_angle(angle)
                        if abs(angle) < self.dynamic_obstacle_angle:
                            print(f"[NAV] 路径阻挡检测: 障碍物在 ({ox:.0f}, {oy:.0f}), "
                                  f"距路径 {dist_to_path:.0f}mm, 距机器人 {dist:.0f}mm")
                            return True
        return False

    def _reached_goal(self, x, y) -> bool:
        if not self.waypoints:
            return False
        gx, gy = self.waypoints[-1]
        return math.hypot(gx - x, gy - y) < self.final_threshold

    def _off_track(self, x, y, theta) -> bool:
        if self.current_wp >= len(self.waypoints):
            return False
        target = self.waypoints[self.current_wp]
        dx = target[0] - x
        dy = target[1] - y
        path_angle = math.atan2(dy, dx)
        perp = abs(math.sin(path_angle - theta) * math.hypot(dx, dy))
        return perp > self.replan_threshold

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle