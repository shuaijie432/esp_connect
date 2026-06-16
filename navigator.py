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

        self.lookahead_min = 600.0
        self.lookahead_max = 1200.0
        self.wp_threshold = 120.0
        self.final_threshold = 60.0

        # 机器人物理尺寸（mm）
        # 底盘支持平移过窄道，膨胀基准用车宽/2（直行通过所需最小半宽），
        # 而非外接圆半径（那是旋转时才需要的）。
        self.robot_width_mm = 270.0           # 27cm 车身宽度
        self.robot_length_mm = 270.0          # 27cm 车身长度
        self.robot_radius_mm = self.robot_width_mm / 2.0   # 135mm
        self.safety_margin_mm = 30.0          # 安全余量（可据实际调整）
        self.total_inflation_mm = self.robot_radius_mm + self.safety_margin_mm  # 165mm

        # 膨胀栅格数 = ceil(total_inflation / resolution)
        # 165/15 = 11 栅格
        # self.obstacle_margin = max(4, int(math.ceil(self.total_inflation_mm / self.mapper.map.resolution)))
        # print(f"[NAV] 膨胀半径: {self.obstacle_margin} 栅格 = {self.obstacle_margin * self.mapper.map.resolution:.0f}mm "
        #       f"(机器人半径{self.robot_radius_mm:.0f}mm + 安全余量{self.safety_margin_mm:.0f}mm)")
        self.obstacle_margin = 4  # 对应 120mm 膨胀
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

        # 规划时冻结的地图（仅保存静态地图快照，膨胀在调用时动态生成）
        self._planned_grid = None
        self._planned_version = -1
        self._plan_frozen = False

        # 速度参数（底盘支持平移）
        self.MAX_VX = 150.0
        self.MAX_VY = 80.0
        self.MAX_VW = 0.5
        self.KP_V = 0.5
        self.KP_W = 1.2

        self._coord_checked = False

        self.target_theta = None
        self._align_settle_until = 0.
        self._alignment_ack_sent = False
        self._send_alignment_ack = False
        self._align_stable_count = 0

        # 障碍物永久融合参数
        self._min_cluster_size = 6
        self._obstacle_stability_threshold = 100.0
        self._obstacle_fusion_interval = 0.5
        self._stable_obstacles = {}
        self._obstacle_id_counter = 0

        # 局部代价地图
        self._local_costmap_size = 80
        self._local_costmap_resolution = 15
        self._local_costmap = np.zeros((self._local_costmap_size, self._local_costmap_size), dtype=np.float32)
        self._local_obstacles = []
        self._local_costmap_center = (0, 0)
        self._local_costmap_valid = False

        # 动态窗口避障参数
        self._dw_enabled = True
        self._dw_safe_distance = 500.0
        self._dw_critical_distance = 300.0
        self._dw_lateral_gain = 1.5

        # 卡住检测参数（长时间卡住才重规划）
        self._stuck_check_start = 0.0
        self._stuck_check_duration = 10.0
        self._stuck_pos_history = deque(maxlen=50)
        self._stuck_dist_threshold = 200.0
        self._total_replan_count = 0
        self._max_replan = 2

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
    # 冻结/解冻规划地图（修改：冻结仅保存静态快照，膨胀在调用时动态生成）
    # ============================================================
    def _freeze_planning_map(self):
        grid = self.mapper.map
        self._planned_grid = grid.log_odds.copy()
        self._planned_version = time.time()
        self._plan_frozen = True
        occ_count = np.count_nonzero(self._planned_grid > grid.occ_thresh)
        print(f"[NAV] 规划地图已冻结，静态占用栅格: {occ_count}")
        # 不再预先计算膨胀，返回动态生成的结果供外部可视化使用
        return self._get_planning_inflated_grid()

    def _unfreeze_planning_map(self):
        self._plan_frozen = False
        self._planned_grid = None
        self._planned_inflated = None  # 兼容旧属性
        self._total_replan_count = 0
        print("[NAV] 规划地图已解冻")

    # ------------------------------------------------------------
    # 新增：从最新点云构建占用掩码（与地图同尺寸）
    # ------------------------------------------------------------
    def _build_pointcloud_occ_mask(self, robot_x, robot_y):
        """返回一个布尔数组，表示实时点云占据的栅格"""
        grid = self.mapper.map
        size = grid.size
        occ_mask = np.zeros((size, size), dtype=bool)

        _, world_pts = self.mapper.get_latest_points()
        if not world_pts:
            return occ_mask

        for wx, wy in world_pts:
            dist = math.hypot(wx - robot_x, wy - robot_y)
            # 忽略自身附近和太远的点，防止雷达噪声和远距离误检
            if dist < 200 or dist > 8000:
                continue

            mx, my = grid.world_to_map(wx, wy)
            if 0 <= mx < size and 0 <= my < size:
                # 不再额外膨胀：全局 obstacle_margin 已经做了统一膨胀，
                # 这里只标记单栅格，避免双重膨胀把通道堵死。
                occ_mask[my, mx] = True
        return occ_mask

    # ------------------------------------------------------------
    # 重写：获取规划用膨胀地图（融合静态地图 + 实时点云）
    # ------------------------------------------------------------
    def _get_planning_inflated_grid(self):
        """返回膨胀后的障碍物网格，综合静态地图与实时点云"""
        grid = self.mapper.map
        robot_x = self.mapper.pose.x
        robot_y = self.mapper.pose.y

        # 静态占用来源：冻结时使用快照，否则使用当前地图
        if self._plan_frozen and self._planned_grid is not None:
            log_odds = self._planned_grid
        else:
            log_odds = grid.log_odds

        # 1. 构建静态占用掩码
        static_occ = log_odds > grid.occ_thresh

        # 2. 构建实时点云占用掩码
        pc_occ = self._build_pointcloud_occ_mask(robot_x, robot_y)

        # 3. 合并两种占用
        combined_occ = static_occ | pc_occ

        # 4. 将机器人所在位置及其周围强制设为可通行（范围需覆盖膨胀半径）
        rx, ry = grid.world_to_map(robot_x, robot_y)
        clear_radius = self.obstacle_margin + 2
        for dx in range(-clear_radius, clear_radius + 1):
            for dy in range(-clear_radius, clear_radius + 1):
                nx, ny = rx + dx, ry + dy
                if 0 <= nx < grid.size and 0 <= ny < grid.size:
                    combined_occ[ny, nx] = False

        # 5. 对合并占用进行膨胀
        try:
            from scipy.ndimage import binary_dilation
            inflated = binary_dilation(combined_occ, iterations=self.obstacle_margin)
        except ImportError:
            # 手动膨胀作为后备
            inflated = combined_occ.copy()
            for _ in range(self.obstacle_margin):
                padded = np.pad(inflated, 1, mode='constant', constant_values=False)
                inflated = (
                    padded[0:-2, 1:-1] |
                    padded[2:, 1:-1] |
                    padded[1:-1, 0:-2] |
                    padded[1:-1, 2:]
                )

        # 膨胀后再次确保机器人栅格可通行
        for dx in range(-clear_radius, clear_radius + 1):
            for dy in range(-clear_radius, clear_radius + 1):
                nx, ny = rx + dx, ry + dy
                if 0 <= nx < grid.size and 0 <= ny < grid.size:
                    inflated[ny, nx] = False

        return inflated

    # ------------------------------------------------------------
    # 保留原 _precompute_inflation 但规划时不再使用，仅用于兼容
    # ------------------------------------------------------------
    def _precompute_inflation(self, grid):
        if self._inflated_grid is not None and self._grid_version == id(grid.log_odds):
            return
        # 仍然预计算以兼容 get_inflated_grid 等旧调用
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

        expired_ids = []
        for obs_id, obs_data in self._stable_obstacles.items():
            if not obs_data.get('is_permanent', False):
                if current_time - obs_data['last_update'] > 30.0:
                    expired_ids.append(obs_id)
            else:
                if current_time - obs_data['last_update'] > 300.0:
                    expired_ids.append(obs_id)

        for obs_id in expired_ids:
            del self._stable_obstacles[obs_id]
            print(f"[OBSTACLE_TRACK] 障碍物 #{obs_id} 已过期")

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

            if best_match_id is not None:
                obs_data = self._stable_obstacles[best_match_id]

                if obs_data.get('is_permanent', False):
                    obs_data['last_update'] = current_time
                    matched_stable_ids.add(best_match_id)
                    permanent_count += 1
                    matched = True

                elif best_match_dist < self._obstacle_stability_threshold:
                    obs_data['is_permanent'] = True
                    obs_data['fixed_position'] = (new_cx, new_cy)
                    obs_data['centroid'] = (new_cx, new_cy)
                    obs_data['cells'] = set(cluster_info['cells'])
                    obs_data['last_update'] = current_time
                    matched_stable_ids.add(best_match_id)
                    stable_count += 1
                    newly_permanent += 1
                    matched = True
                    print(f"[OBSTACLE_TRACK] 障碍物 #{best_match_id} 位置稳定 "
                          f"({best_match_dist:.0f}mm < {self._obstacle_stability_threshold}mm) "
                          f"→ 永久固化到地图！")

                else:
                    obs_data['centroid'] = (new_cx, new_cy)
                    obs_data['cells'] = set(cluster_info['cells'])
                    obs_data['last_update'] = current_time
                    matched_stable_ids.add(best_match_id)
                    updated_count += 1
                    matched = True
                    print(f"[OBSTACLE_TRACK] 障碍物 #{best_match_id} 位置更新 "
                          f"→ ({new_cx:.0f},{new_cy:.0f}) "
                          f"距离={best_match_dist:.0f}mm，继续观察")

            if not matched:
                self._obstacle_id_counter += 1
                new_id = self._obstacle_id_counter
                self._stable_obstacles[new_id] = {
                    'centroid': (new_cx, new_cy),
                    'cells': set(cluster_info['cells']),
                    'last_update': current_time,
                    'is_permanent': False,
                    'fixed_position': None
                }
                matched_stable_ids.add(new_id)
                updated_count += 1
                print(f"[OBSTACLE_TRACK] 新大障碍物 #{new_id} @({new_cx:.0f},{new_cy:.0f}) "
                      f"大小={cluster_info['size']}栅格({cluster_info['size_mm']:.0f}mm)")

        # 如果规划未冻结，将永久障碍物写入静态地图（后续规划会用到）
        if not self._plan_frozen:
            all_cells_to_mark = set()
            for obs_id, obs_data in self._stable_obstacles.items():
                if obs_data.get('is_permanent', False):
                    all_cells_to_mark.update(obs_data['cells'])

            added = 0
            for mx, my in all_cells_to_mark:
                if 0 <= mx < grid.size and 0 <= my < grid.size:
                    grid.log_odds[my, mx] = grid.occ_thresh + 5.0
                    added += 1

            if added > 0:
                self._inflated_grid = None
                self._grid_version = -1
        else:
            added = 0
            print(f"[OBSTACLE_TRACK] 规划已冻结，跳过地图更新，记录 {len(matched_stable_ids)} 个障碍物")

        return {
            'added': added,
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
        self._grid_version = -1
        self._unfreeze_planning_map()
        print("[NAV] 所有动态障碍物已清除，规划地图已解冻")

    # ============================================================
    # 局部代价地图（保持不变）
    # ============================================================
    def _update_local_costmap(self, robot_x, robot_y, robot_theta):
        grid = self.mapper.map
        size = self._local_costmap_size
        half = size // 2
        res = self._local_costmap_resolution

        self._local_costmap.fill(0.0)

        for obs_id, obs_data in self._stable_obstacles.items():
            if not obs_data.get('is_permanent', False):
                continue
            for mx, my in obs_data['cells']:
                wx, wy = grid.map_to_world(mx, my)
                dx = wx - robot_x
                dy = wy - robot_y
                local_x = dx * math.cos(robot_theta) + dy * math.sin(robot_theta)
                local_y = -dx * math.sin(robot_theta) + dy * math.cos(robot_theta)
                lx = int(local_x / res + half)
                ly = int(local_y / res + half)
                if 0 <= lx < size and 0 <= ly < size:
                    self._local_costmap[ly, lx] = 100.0

        _, world_pts = self.mapper.get_latest_points()
        self._local_obstacles = []

        for wx, wy in world_pts:
            dx = wx - robot_x
            dy = wy - robot_y
            dist = math.hypot(dx, dy)
            if dist > 1500:
                continue

            local_x = dx * math.cos(robot_theta) + dy * math.sin(robot_theta)
            local_y = -dx * math.sin(robot_theta) + dy * math.cos(robot_theta)

            if local_x < -100:
                continue

            self._local_obstacles.append((local_x, local_y, dist))

            lx = int(local_x / res + half)
            ly = int(local_y / res + half)
            if 0 <= lx < size and 0 <= ly < size:
                cost = min(100.0, 5000.0 / max(dist, 50))
                self._local_costmap[ly, lx] = max(self._local_costmap[ly, lx], cost)

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
    # 动态窗口避障（修复狭窄通道振荡问题）
    # ============================================================
    def _dynamic_window_avoidance(self, x, y, theta, base_vx, base_vy, base_vw) -> Tuple[float, float, float]:
        self._update_local_costmap(x, y, theta)

        front_obstacles = []
        left_obstacles = []
        right_obstacles = []

        for lx, ly, dist in self._local_obstacles:
            angle = math.atan2(ly, lx)
            if abs(angle) < math.radians(20) and dist < 600:
                front_obstacles.append((lx, ly, dist))
            elif math.radians(20) <= angle < math.radians(60) and dist < 500:
                left_obstacles.append((lx, ly, dist))
            elif math.radians(-60) < angle <= math.radians(-20) and dist < 500:
                right_obstacles.append((lx, ly, dist))

        front_min = min((d for _, _, d in front_obstacles), default=float('inf'))
        left_min = min((d for _, _, d in left_obstacles), default=float('inf'))
        right_min = min((d for _, _, d in right_obstacles), default=float('inf'))

        vx, vy, vw = base_vx, base_vy, base_vw

        path_dir = self._get_path_direction(x, y, theta)
        if path_dir is None:
            path_dir = 0.0

        # 计算路径建议的侧向趋势（正=左，负=右）
        path_lateral = math.sin(path_dir)

        # 计算通道宽度，用于抑制狭窄通道中的过度 lateral 振荡
        channel_width = float('inf')
        if left_min < float('inf') and right_min < float('inf'):
            channel_width = left_min + right_min

        # 紧急避障：前方极近 -> 停止前进，优先侧向滑出，不要冲进去
        if front_min < self._dw_critical_distance:
            print(f"[DW] 极近！前方{front_min:.0f}mm，停止前进，侧向避让")
            vx = 0.0  # 停止前进，避免冲进 EMERGENCY 触发后退振荡
            # 优先遵循全局路径方向，同时检查该侧是否有足够空间
            if path_lateral > 0.3 and left_min > 250:
                vy = base_vy + 60.0   # 路径偏左，且左侧有空间
            elif path_lateral < -0.3 and right_min > 250:
                vy = base_vy - 60.0  # 路径偏右，且右侧有空间
            elif right_min > left_min + 150 and right_min > 250:
                vy = base_vy - 40.0  # 右侧空间明显更大
            elif left_min > right_min + 150 and left_min > 250:
                vy = base_vy + 40.0   # 左侧空间明显更大
            else:
                vy = base_vy * 0.5  # 不确定时保持原侧向趋势但减速
            vw = 0.0

        # 减速避障
        elif front_min < self._dw_safe_distance:
            scale = max(0.2, (front_min - self._dw_critical_distance) /
                       (self._dw_safe_distance - self._dw_critical_distance))
            vx = base_vx * scale

            # 根据通道宽度限制 lateral_delta，避免在窄道中左右剧烈摇摆
            lateral_delta = self._dw_lateral_gain * (self._dw_safe_distance - front_min) * 0.3
            if channel_width < 1200:
                # 通道越窄，lateral 避让越弱，优先相信全局路径
                channel_factor = max(0.15, channel_width / 1200.0)
                lateral_delta *= channel_factor

            # 侧向修正限制：避免单次大幅横跳导致 oscillation
            max_lateral_cmd = 40.0 if channel_width < 1000 else 70.0

            if right_min > left_min + 80 and right_min > 300:
                # 右侧空间更大，倾向向右，但用 path_bias 修正
                path_bias = math.sin(path_dir) * 40  # 路径偏左时(>0) 减少向右趋势
                vy = base_vy - min(lateral_delta, max_lateral_cmd) + path_bias
                print(f"[DW] 前方{front_min:.0f}mm，减速+向右避让 vy={vy:.0f} path_bias={path_bias:.0f}")
            elif left_min > right_min + 80 and left_min > 300:
                # 左侧空间更大，倾向向左，但用 path_bias 修正
                path_bias = math.sin(path_dir) * 40  # 路径偏右时(<0) 减少向左趋势
                vy = base_vy + min(lateral_delta, max_lateral_cmd) + path_bias
                print(f"[DW] 前方{front_min:.0f}mm，减速+向左避让 vy={vy:.0f} path_bias={path_bias:.0f}")
            else:
                # 两侧都近，大幅减速，优先回归路径
                vx = base_vx * 0.2
                vy = base_vy + path_lateral * 60
                print(f"[DW] 前方{front_min:.0f}mm，两侧狭窄，沿路径侧向回归 vy={vy:.0f}")

        else:
            # 前方安全，但两侧有障碍时微调保持居中
            if left_min < 400 and right_min > 500:
                vy = base_vy - 15.0  # 离左侧太近，稍微右移
            elif right_min < 400 and left_min > 500:
                vy = base_vy + 15.0  # 离右侧太近，稍微左移

        vx = max(0.0, min(self.MAX_VX, vx))
        vy = max(-self.MAX_VY, min(self.MAX_VY, vy))
        vw = max(-self.MAX_VW, min(self.MAX_VW, vw))

        return vx, vy, vw

    # ============================================================
    # 导航主循环（已移除 _update_map_with_pointcloud 调用）
    # ============================================================
    def update(self, x: float, y: float, theta: float) -> Tuple[float, float, float]:
        if self.state in ("IDLE", "FAILED", "DONE"):
            return 0.0, 0.0, 0.0

        is_start_phase = (self.current_wp < 3) and (len(self.waypoints) > 3)

        if self.state == "ALIGNING":
            return self._align_heading_step(x, y, theta)

        if self._reached_goal(x, y):
            if self.target_theta is not None:
                self.state = "ALIGNING"
                print(f"[NAV] 到达目标位置，开始对准角度 {math.degrees(self.target_theta):.1f}°")
                return 0.0, 0.0, 0.0
            else:
                self.state = "DONE"
                print("[NAV] 到达目标！")
                return 0.0, 0.0, 0.0

        now = time.time()

        # 卡住检测（长时间卡住才重规划）
        self._stuck_pos_history.append((x, y, now))

        if self._stuck_check_start == 0.0:
            self._stuck_check_start = now

        if len(self._stuck_pos_history) >= 10 and now - self._stuck_check_start > self._stuck_check_duration:
            old_x, old_y, old_t = self._stuck_pos_history[0]
            if now - old_t > self._stuck_check_duration:
                moved = math.hypot(x - old_x, y - old_y)
                if moved < self._stuck_dist_threshold and self._total_replan_count < self._max_replan:
                    print(f"[NAV] 卡住检测！{self._stuck_check_duration:.0f}秒内只移动了 {moved:.0f}mm，触发重规划 (第{self._total_replan_count+1}次)")
                    self._unfreeze_planning_map()
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

        # 瞬时卡住恢复
        if not is_start_phase and self.state not in ("ALIGNING", "EVADE_EMERGENCY", "AVOIDING") and self.last_pos:
            moved = math.hypot(x - self.last_pos[0], y - self.last_pos[1])
            if moved > 30:
                self.stuck_timer = now
            elif now - self.stuck_timer > 2.0 and now > self.stuck_recovery_until:
                print("[NAV] 瞬时卡住检测触发！执行后退恢复...")
                self.state = "STUCK_RECOVERY"
                self.stuck_recovery_until = now + 0.8
                self.stuck_timer = now
                return -100.0, 0.0, 0.0
        self.last_pos = (x, y)

        if now < self.stuck_recovery_until:
            return -100.0, 0.0, 0.0

        if self.state == "STUCK_RECOVERY" and now >= self.stuck_recovery_until:
            print("[NAV] 后退完成...")
            self.state = "FOLLOWING"
            return 0.0, 0.0, 0.0

        self._sync_waypoint_index(x, y)

        # 横向偏差过大 -> 重规划
        lateral_error = self._calc_lateral_error(x, y, theta)
        if lateral_error > 400.0 and now - self.last_replan_time > 5.0:
            print(f"[NAV] 横向偏差 {lateral_error:.0f}mm 持续过大，触发重规划")
            self._unfreeze_planning_map()
            if self.waypoints:
                if self.set_target(self.waypoints[-1][0], self.waypoints[-1][1], self.target_theta):
                    self.last_replan_time = now
                    return 0.0, 0.0, 0.0
        elif lateral_error > 300.0:
            print(f"[NAV] 横向偏差 {lateral_error:.0f}mm 较大，依赖回归修正")

        if self._off_track(x, y, theta):
            print("[NAV] 偏离航线，依赖回归修正")

        obstacle_level = self._check_obstacle_level(x, y, theta)
        front_dist = self._get_front_distance(x, y, theta)

        # 障碍物附近重置卡住计时器：靠近障碍物时移动慢是正常的，不要触发后退振荡
        if front_dist < 600.0 or obstacle_level != "CLEAR":
            self.stuck_timer = now

        # EMERGENCY: 前方极近，强制接管
        if not is_start_phase and obstacle_level == "EMERGENCY":
            vx, vy, vw = self._evade_obstacle(x, y, theta)
            self.state = "EVADE_EMERGENCY"
            return vx, vy, vw

        self.state = "FOLLOWING"
        base_vx, base_vy, base_vw = self._pure_pursuit_step(x, y, theta)

        # 安全减速（基于实时点云）
        if front_dist < 800.0:
            speed_scale = max(0.15, (front_dist - 300.0) / 500.0)
            base_vx = base_vx * speed_scale
            base_vy = base_vy * speed_scale

        if self._dw_enabled and not is_start_phase and obstacle_level != "CLEAR":
            vx, vy, vw = self._dynamic_window_avoidance(x, y, theta, base_vx, base_vy, base_vw)
        else:
            vx, vy, vw = base_vx, base_vy, base_vw

        if is_start_phase:
            if vx < 50.0:
                vx = 80.0
            if abs(vy) > vx * 0.5:
                vy = math.copysign(vx * 0.5, vy) if vy != 0 else 0.0

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

    def _get_front_distance(self, x, y, theta) -> float:
        _, world_pts = self.mapper.get_latest_points()
        if not world_pts:
            return float('inf')
        front_min = float('inf')
        for wx, wy in world_pts:
            dx = wx - x
            dy = wy - y
            dist = math.hypot(dx, dy)
            if dist > 2000 or dist < 100:
                continue
            angle = self._normalize_angle(math.atan2(dy, dx) - theta)
            if abs(angle) < math.radians(30):
                front_min = min(front_min, dist)
        return front_min

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
        self.waypoints = self._smooth_path(raw_path)
        self.current_wp = 0

        self.state = "FOLLOWING"
        self.last_replan_time = time.time()
        self.stuck_timer = time.time()
        self.last_pos = (self.mapper.pose.x, self.mapper.pose.y)

        self._stuck_check_start = time.time()
        self._stuck_pos_history.clear()

        self._align_settle_until = 0.0
        self._alignment_ack_sent = False
        self._send_alignment_ack = False
        self._align_stable_count = 0

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
        self._unfreeze_planning_map()
        print("[NAV] 导航已取消，规划地图已解冻")

    def _align_heading_step(self, x: float, y: float, theta: float) -> Tuple[float, float, float]:
        angle_diff = self._normalize_angle(self.target_theta - theta)
        abs_diff = abs(angle_diff)
        now = time.time()

        if abs_diff >= math.radians(3.0):
            self._align_settle_until = 0.0
            self._alignment_ack_sent = False
            self._align_stable_count = 0
            vw_raw = self.KP_W * angle_diff * 1.0
            vw = max(-0.4, min(0.4, vw_raw))
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
    # Pure Pursuit + 路径回归修正（增强版）
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

        if is_final and dist < 800:
            scale = max(0.15, dist / 800.0)
            max_vx *= scale
            max_vy *= scale
            max_vw *= scale

        kp_v = self.KP_V
        if is_final and dist < 400:
            kp_v = 0.25

        # 狭窄通道检测：两侧都有障碍物时自动降速
        _, world_pts = self.mapper.get_latest_points()
        channel_width = float('inf')
        if world_pts:
            left_wall = float('inf')
            right_wall = float('inf')
            for wx, wy in world_pts:
                dx = wx - x
                dy = wy - y
                d = math.hypot(dx, dy)
                if d > 1500 or d < 100:
                    continue
                angle = self._normalize_angle(math.atan2(dy, dx) - theta)
                a_deg = math.degrees(angle)
                if 30 <= a_deg < 90:
                    left_wall = min(left_wall, d)
                elif -90 < a_deg <= -30:
                    right_wall = min(right_wall, d)
            if left_wall < float('inf') and right_wall < float('inf'):
                channel_width = left_wall + right_wall
                if channel_width < 1200:
                    channel_scale = max(0.3, channel_width / 1200.0)
                    max_vx *= channel_scale
                    max_vy *= channel_scale
                    if channel_width < 800:
                        print(f"[CHANNEL] 狭窄通道 {channel_width:.0f}mm，降速至 {channel_scale:.1f}")

        if abs_angle < math.radians(75):
            vx_raw = kp_v * local_x
        else:
            vx_raw = 0.0

        vy_raw = self.KP_V * local_y * 0.8

        v_mag = math.hypot(vx_raw, vy_raw)
        if v_mag > max_vx:
            ratio = max_vx / v_mag
            vx_raw *= ratio
            vy_raw *= ratio

        vx = max(0.0, min(max_vx, vx_raw))
        vy = max(-max_vy, min(max_vy, vy_raw))

        # 增强的路径回归约束（在狭窄通道中权重更高）
        path_lateral_err, path_proj = self._calc_path_projection(x, y)
        regress_gain = 3.0 if channel_width < 900 else 2.5
        if abs(path_lateral_err) > 10.0 and len(self.waypoints) > 2:
            reg_dx = path_proj[0] - x
            reg_dy = path_proj[1] - y
            reg_dist = math.hypot(reg_dx, reg_dy)
            if reg_dist > 5:
                vy_correction = -path_lateral_err * regress_gain
                vy_correction = max(-self.MAX_VY, min(self.MAX_VY, vy_correction))
                vy += vy_correction

                reg_local_x = reg_dx * cos_t + reg_dy * sin_t
                if reg_local_x > 0:
                    vx += min(50.0, reg_local_x * 0.5)

                print(f"[REGRESS] 横向偏差 {path_lateral_err:+.1f}mm，vy修正 {vy_correction:.0f} -> vy={vy:.0f}")

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
        vw_err = angle_diff * 0.7 + path_angle_diff * 0.3
        vw = max(-max_vw, min(max_vw, self.KP_W * vw_err))

        if abs_angle > math.radians(60):
            vx *= 0.3
            print(f"[TURN] 角度差 {math.degrees(abs_angle):.1f}°，减速转向")

        return vx, vy, vw

    def _check_obstacle_level(self, x, y, theta) -> str:
        _, world_pts = self.mapper.get_latest_points()
        if not world_pts:
            return "CLEAR"

        front_min = float('inf')
        left_min = float('inf')
        right_min = float('inf')

        for wx, wy in world_pts:
            dx = wx - x
            dy = wy - y
            dist = math.hypot(dx, dy)
            if dist > 1000:
                continue
            angle = self._normalize_angle(math.atan2(dy, dx) - theta)

            if abs(angle) < math.radians(25):
                front_min = min(front_min, dist)
            elif math.radians(25) <= angle < math.radians(70):
                left_min = min(left_min, dist)
            elif math.radians(-70) < angle <= math.radians(-25):
                right_min = min(right_min, dist)

        if front_min < 250:
            return "EMERGENCY"
        if front_min < 450 or left_min < 350 or right_min < 350:
            return "CAUTION"

        return "CLEAR"

    def _evade_obstacle(self, x, y, theta) -> Tuple[float, float, float]:
        _, world_pts = self.mapper.get_latest_points()
        if not world_pts:
            return 0.0, 0.0, 0.0

        path_dir = self._get_path_direction(x, y, theta)
        if path_dir is None:
            path_dir = 0.0

        # 计算各方向最近障碍物距离（更大角度范围）
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

        # 如果前方还有足够空间，不需要紧急接管，回到路径跟踪
        if front_min > 500 and fleft_min > 400 and fright_min > 400:
            base_vx, base_vy, base_vw = self._pure_pursuit_step(x, y, theta)
            return base_vx * 0.4, base_vy * 0.4, base_vw * 0.4

        print(f"[EVADE] 前方{front_min:.0f}mm 左前{fleft_min:.0f}mm 右前{fright_min:.0f}mm，路径方向={math.degrees(path_dir):.0f}°")

        # 基于路径方向搜索最优安全方向
        best_dir = None
        best_score = -float('inf')

        # 优先测试路径方向及附近方向
        # 当已经非常靠近障碍物时，优先搜索侧向方向（左右横移）而不是正前方
        test_dirs = []
        if front_min < 300:
            # 极近时优先侧向：先路径侧向，再斜向，最后正前
            for priority_offset in [90, -90, 75, -75, 60, -60, 45, -45, 30, -30, 15, -15, 0, 120, -120]:
                test_dirs.append(self._normalize_angle(path_dir + math.radians(priority_offset)))
        else:
            for priority_offset in [0, 15, -15, 30, -30, 45, -45, 60, -60, 75, -75, 90, -90, 120, -120]:
                test_dirs.append(self._normalize_angle(path_dir + math.radians(priority_offset)))

        # 检查前方多个距离点，近距离即可，不要要求700mm都clear（窄通道做不到）
        check_dists = [200, 400]
        # 安全半径略放宽，因为全局路径已经做过膨胀了
        safe_radius = max(120.0, self.total_inflation_mm * 0.85)

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

            # 评分：越接近路径方向越好，障碍物越远越好
            dir_diff = abs(self._normalize_angle(test_rad - path_dir))
            score = 200 - math.degrees(dir_diff) * 1.2 + min_obstacle_dist * 0.05

            # 前方极近时，强烈偏好有明显侧向分量的方向（左右滑出）
            if front_min < 300:
                side_component = abs(math.sin(test_rad))
                score += side_component * 80  # 强烈奖励侧向
            elif front_min < 450:
                if abs(math.sin(test_rad)) > 0.5:
                    score += 30

            if score > best_score:
                best_score = score
                best_dir = test_rad

        if best_dir is not None:
            # 根据前方距离决定速度
            if front_min < 300:
                speed = 40.0
            elif front_min < 500:
                speed = 60.0
            else:
                speed = 80.0

            vx = speed * math.cos(best_dir)
            vy = speed * math.sin(best_dir)
            if vx < 0:
                vx = 0.0
            vw = 0.0
            print(f"[EVADE] 选择方向={math.degrees(best_dir):.0f}° vx={vx:.0f} vy={vy:.0f}")
            return vx, vy, vw

        # 所有方向都不安全，慢速后退作为最后手段
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
    # A* 和膨胀函数（使用动态融合地图）
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

        clear_r = self.obstacle_margin + 2
        for dx in range(-clear_r, clear_r + 1):
            for dy in range(-clear_r, clear_r + 1):
                nx, ny = sx + dx, sy + dy
                if in_bounds(nx, ny):
                    inflated[ny, nx] = False

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

                occ_penalty = 0.0
                for ox in range(-3, 4):
                    for oy in range(-3, 4):
                        if in_bounds(nx + ox, ny + oy):
                            if inflated[ny + oy, nx + ox]:
                                dist = math.hypot(ox, oy)
                                if dist < 0.01:
                                    penalty = 2.0
                                else:
                                    penalty = 1.0 / (dist + 0.5)
                                occ_penalty += penalty

                tentative = g_score[cy, cx] + cost + occ_penalty
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

    def _smooth_path(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if len(path) <= 2:
            return path

        sampled = [path[0]]
        for p in path[1:-1]:
            if math.hypot(p[0] - sampled[-1][0], p[1] - sampled[-1][1]) > 120:
                sampled.append(p)
        sampled.append(path[-1])

        smooth = [list(p) for p in sampled]
        for _ in range(5):
            for i in range(1, len(smooth) - 1):
                ax = (smooth[i-1][0] + smooth[i+1][0]) * 0.5
                ay = (smooth[i-1][1] + smooth[i+1][1]) * 0.5
                smooth[i][0] = 0.7 * smooth[i][0] + 0.3 * ax
                smooth[i][1] = 0.7 * smooth[i][1] + 0.3 * ay

        return [(p[0], p[1]) for p in smooth]

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