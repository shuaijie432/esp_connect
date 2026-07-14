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
        # 底盘支持平移过窄道，膨胀基准用车宽/2（直行通过所需最小半宽），
        # 而非外接圆半径（那是旋转时才需要的）。


        self.robot_width_mm = 250.0           # 27cm 车身宽度
        self.robot_length_mm = 250.0          # 27cm 车身长度
        self.robot_radius_mm = self.robot_width_mm / 2.0   # 125mm
        self.safety_margin_mm = 50.0           # DWA 额外安全余量
        self.total_inflation_mm = self.robot_radius_mm + self.safety_margin_mm  # 175mm

        # ★ A* 硬膨胀 = 16格 = 160mm（略小于 DWA 碰撞半宽 175mm，由代价梯度补足）
        # DWA碰撞半宽 = robot_radius(125) + channel_margin(50) = 175mm
        self.obstacle_margin = 16

        # ★ DWA 碰撞框参数
        # 碰撞半宽 = robot_radius(125) + channel_margin = 175mm，框宽350mm
        #   修改这个值影响 DWA 碰撞框 / 红色虚线圆 / 路径堵塞检测
        self.channel_margin_mm = 50.0   # 碰撞半宽=125+50=175mm
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
        self._cost_grid = None          # 距离变换代价网格（A* 根据离障碍物距离计算代价）

        # 规划时冻结的地图（仅保存静态地图快照，膨胀在调用时动态生成）
        self._planned_grid = None
        self._planned_version = -1
        self._plan_frozen = False

        # 速度参数（底盘支持平移）
        self.MAX_VX = 150.0   # 小地图降速，窄道中留更多反应时间
        self.MAX_VY = 80.0    # 侧移速度降低
        self.MAX_VW = 0.4     # 角速度提高，小空间转弯更灵活
        self.KP_V = 0.5
        self.KP_W = 1.4       # 提高角度增益，对方向偏差反应更积极

        self._coord_checked = False

        self.target_theta = None
        self._align_settle_until = 0.
        self._alignment_ack_sent = False
        self._send_alignment_ack = False
        self._align_stable_count = 0

        # 障碍物永久融合参数
        self._min_cluster_size = 1                               # 至少3个连成片的栅格才算障碍物（滤除噪点）
        self._permanent_after_sec = 1.5                          # 连续观测超过1.5秒 → 写入静态地图
        self._expire_after_sec = 10.0                            # 超过10秒未扫到 → 从静态地图删除
        self._obstacle_cost_gain = 60.0                          # A*动态障碍物代价增益：强力推开路径（15→60）
        self._obstacle_cost_decay = 5.0                          # A*代价衰减（栅格），越小惩罚越集中→梯度越陡（8→5）
        self._static_cost_gain = 80.0                            # 静态墙壁额外代价：强力推向走廊中心（30→80）
        self._static_cost_decay = 6.0                            # 静态墙壁代价衰减，更陡的梯度（12→6）
        self._static_cost_grid = None                            # 静态地图距离变换（独立于动态障碍物）
        self._obstacle_fusion_interval = 0.5
        self._stable_obstacles = {}
        self._obstacle_id_counter = 0

        # 局部代价地图
        self._local_costmap_size = 80
        self._local_costmap_resolution = 10   # 与 RESOLUTION_MM 保持一致
        self._local_costmap = np.zeros((self._local_costmap_size, self._local_costmap_size), dtype=np.float32)
        self._local_obstacles = []
        self._local_costmap_center = (0, 0)
        self._local_costmap_valid = False

        # 障碍物短期记忆：DWA 避障时记住最近几秒出现过的障碍物
        # 即使当前帧没扫到也不立即"忘记"，避免雷达漏帧导致撞上
        self._local_obstacle_memory = {}       # {(grid_x, grid_y): (wx, wy, last_seen_time)}
        self._local_obstacle_memory_ttl = 15  # 记忆保留 15 秒，与追踪器 TTL 配合使用

        # 动态窗口避障参数
        self._dw_enabled = True
        # 四周统一安全距离：车头/车尾/侧方都使用同样的阈值
        self._uniform_clearance = 200.0   # DWA安全余量 mm（提高：150→200）
        self._dw_safe_distance = 150.0    # 前方150mm开始减速（提高：100→150）
        self._dw_critical_distance = 100.0  # 前方100mm停止（提高：70→100）
        self._dw_lateral_gain = 2.0       # 侧向避障增益（提高：1.7→2.0）

        # 卡住检测参数（长时间卡住才重规划）
        self._stuck_check_start = 0.0
        self._stuck_check_duration = 5.0
        self._stuck_pos_history = deque(maxlen=50)
        self._stuck_dist_threshold = 200.0
        self._total_replan_count = 0
        self._max_replan = 999

        # 起点保护：首次导航先向右上方移动 3 帧，
        # 后续导航先原地旋转对齐朝向，再开始路径跟踪
        # 1=对齐中, 2=完成（进入正常路径跟踪）
        self._startup_align_phase = 2
        self._startup_align_stable = 0
        self._nav_count = 0  # 导航次数：首次=起点保护，后续=原地旋转对齐
        self.safety_boost = 0.0  # 安全余量加成(mm)，特定导航点可临时增大

        # 检查 scipy 是否可用（binary_dilation 和 distance_transform_edt 依赖它）
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
    # 冻结/解冻规划地图（修改：冻结仅保存静态快照，膨胀在调用时动态生成）
    # ============================================================
    def _freeze_planning_map(self):
        # ★ 已冻结时不再重拍快照：整个导航过程用同一张地图，
        #   避免里程计漂移后新快照与机器人位置对不上 → 点云"漂移"
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
            if dist < 80 or dist > 8000:
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

        # 3. 合并静态占用 + 实时点云
        combined_occ = static_occ | pc_occ

        # 3.5 将所有追踪中的障碍物栅格（含未固化的）叠加到占用掩码
        #     不再写入静态地图，障碍物过期后自然消失
        for obs_id, obs_data in self._stable_obstacles.items():
            for mx, my in obs_data['cells']:
                if 0 <= mx < grid.size and 0 <= my < grid.size:
                    combined_occ[my, mx] = True

        # 4. 膨胀前清空：将机器人周围障碍物从临时掩码中移除
        # 清空半径 = obstacle_margin，覆盖 DWA 碰撞半宽内的全部障碍物
        # 只影响本次 A* 的临时膨胀，不修改静态地图
        rx, ry = grid.world_to_map(robot_x, robot_y)
        pre_clear = self.obstacle_margin  # 16格=160mm
        for dx in range(-pre_clear, pre_clear + 1):
            for dy in range(-pre_clear, pre_clear + 1):
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

        # 膨胀后清空：确保 A* 有足够空间从起点扩展（太小会导致无路径）
        post_clear = max(self.obstacle_margin // 2, 10)  # 10格=100mm，A* 起步空间充足
        for dx in range(-post_clear, post_clear + 1):
            for dy in range(-post_clear, post_clear + 1):
                nx, ny = rx + dx, ry + dy
                if 0 <= nx < grid.size and 0 <= ny < grid.size:
                    inflated[ny, nx] = False

        # 6. 距离变换：计算每个栅格到最近障碍物的欧氏距离（栅格单位）
        #    用于 A* 代价函数，让路径自然远离障碍物，给 DWA 留出绕行空间
        try:
            from scipy.ndimage import distance_transform_edt
            self._cost_grid = distance_transform_edt(~combined_occ)
            # 静态墙壁独立代价：指数衰减，把路径推向走廊中心
            self._static_cost_grid = distance_transform_edt(~static_occ)
        except ImportError:
            self._cost_grid = None
            self._static_cost_grid = None

        return inflated

    def _get_cost_grid(self):
        """返回距离变换代价网格，供 A* 使用。
        每个栅格的值 = 到最近障碍物的栅格距离（欧氏距离）。
        需要先调用 _get_planning_inflated_grid() 来更新。
        """
        return self._cost_grid

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

        grid = self.mapper.map
        expired_ids = []
        for obs_id, obs_data in self._stable_obstacles.items():
            if current_time - obs_data['last_update'] > self._expire_after_sec:
                expired_ids.append(obs_id)

        for obs_id in expired_ids:
            obs_data = self._stable_obstacles[obs_id]
            if obs_data.get('is_permanent', False):
                for mx, my in obs_data['cells']:
                    if 0 <= mx < grid.size and 0 <= my < grid.size:
                        grid.log_odds[my, mx] = 0.0
            del self._stable_obstacles[obs_id]

        # ---- 数量上限：防止长期运行中追踪障碍物无限累积 ----
        MAX_TRACKED = 30
        if len(self._stable_obstacles) > MAX_TRACKED:
            # 优先驱逐最旧的非永久障碍物，其次驱逐最旧的永久障碍物
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
                            grid.log_odds[my, mx] = 0.0
                del self._stable_obstacles[obs_id]

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

            if best_match_id is not None and best_match_dist < 500.0:  # 最大匹配距离500mm
                obs_data = self._stable_obstacles[best_match_id]

                if obs_data.get('is_permanent', False):
                    # 已固化：只需更新时间戳
                    obs_data['last_update'] = current_time
                    matched_stable_ids.add(best_match_id)
                    permanent_count += 1
                    matched = True

                else:
                    # 未固化：更新位置和栅格，检查是否已连续观测超过3秒
                    obs_data['centroid'] = (new_cx, new_cy)
                    obs_data['cells'] = set(cluster_info['cells'])
                    obs_data['last_update'] = current_time
                    matched_stable_ids.add(best_match_id)
                    updated_count += 1
                    matched = True

                    elapsed = current_time - obs_data['first_seen']
                    if elapsed >= self._permanent_after_sec:
                        # 连续观测超过3秒 → 永久固化到静态地图
                        obs_data['is_permanent'] = True
                        obs_data['fixed_position'] = (new_cx, new_cy)
                        for mx, my in obs_data['cells']:
                            if 0 <= mx < grid.size and 0 <= my < grid.size:
                                grid.log_odds[my, mx] = grid.log_occ  # +6.0 = 占用
                        stable_count += 1
                        newly_permanent += 1
                    else:
                        pass  # 静默

            if not matched:
                self._obstacle_id_counter += 1
                new_id = self._obstacle_id_counter
                self._stable_obstacles[new_id] = {
                    'centroid': (new_cx, new_cy),
                    'cells': set(cluster_info['cells']),
                    'first_seen': current_time,      # 首次观测时间
                    'last_update': current_time,
                    'is_permanent': False,
                    'fixed_position': None
                }
                matched_stable_ids.add(new_id)
                updated_count += 1

        # 跟踪的障碍物动态叠加到规划和避障中（不区分是否永久）
        # 已永久固化的障碍物已写入 static map，这里再叠加一次确保不遗漏
        all_permanent_cells = set()
        for obs_id, obs_data in self._stable_obstacles.items():
            if obs_data.get('is_permanent', False):
                all_permanent_cells.update(obs_data['cells'])

        # 标记需要重规划（有新障碍物出现或障碍物状态变化）
        need_replan = (updated_count > 0 or newly_permanent > 0) and self._plan_frozen

        return {
            'added': 0,  # 不再写入静态地图，始终为 0
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
    # 局部代价地图（含障碍物短期记忆）
    # ============================================================
    def _update_local_costmap(self, robot_x, robot_y, robot_theta):
        grid = self.mapper.map
        size = self._local_costmap_size
        half = size // 2
        res = self._local_costmap_resolution

        self._local_costmap.fill(0.0)

        # 将所有追踪中的障碍物加入局部代价地图，同时收集 DWA 碰撞检测数据
        # 已固化的 = 高置信度(代价100)，未固化的 = 中置信度(代价50)
        _tracked_for_dwa = []  # (local_x, local_y, dist, key) 延迟追加到 _local_obstacles
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
                # 代价地图
                lx = int(local_x / res + half)
                ly = int(local_y / res + half)
                if 0 <= lx < size and 0 <= ly < size:
                    self._local_costmap[ly, lx] = max(self._local_costmap[ly, lx], cost)
                # 收集 DWA 碰撞检测数据（延迟去重追加）
                if dist <= 2000:
                    key = (round(wx / 50.0), round(wy / 50.0))
                    _tracked_for_dwa.append((local_x, local_y, dist, key))

        _, world_pts = self.mapper.get_latest_points()
        self._local_obstacles = []
        now = time.time()

        # ---- 更新障碍物短期记忆 ----
        # 将当前帧的障碍物按 50mm 网格合并，记录时间戳
        current_keys = set()
        for wx, wy in world_pts:
            dx = wx - robot_x
            dy = wy - robot_y
            dist = math.hypot(dx, dy)
            if dist > 1500 or dist < 20:  # 只过滤 20mm 内的自身反射，保留近距离障碍物
                continue
            key = (round(wx / 50.0), round(wy / 50.0))
            current_keys.add(key)
            self._local_obstacle_memory[key] = (wx, wy, now)

        # 清除过期的记忆（超过 TTL 未再出现的障碍物）
        expired = [k for k, v in self._local_obstacle_memory.items()
                   if now - v[2] > self._local_obstacle_memory_ttl]
        for k in expired:
            del self._local_obstacle_memory[k]

        # ---- 将记忆中的障碍物加入局部列表和代价地图 ----
        for key, (wx, wy, t) in self._local_obstacle_memory.items():
            dx = wx - robot_x
            dy = wy - robot_y
            dist = math.hypot(dx, dy)
            if dist > 1500:
                continue

            local_x = dx * math.cos(robot_theta) + dy * math.sin(robot_theta)
            local_y = -dx * math.sin(robot_theta) + dy * math.cos(robot_theta)

            # 360° 全向感知：不按方向过滤，车尾方向也要检测

            age = now - t
            # 当前帧的点：完整代价；记忆中的点：代价随时间衰减
            if key in current_keys:
                confidence = 1.0
            else:
                # 记忆点随年龄衰减：0~TTL 秒内线性降到 0.3
                confidence = max(0.3, 1.0 - age / self._local_obstacle_memory_ttl)

            self._local_obstacles.append((local_x, local_y, dist))

            lx = int(local_x / res + half)
            ly = int(local_y / res + half)
            if 0 <= lx < size and 0 <= ly < size:
                cost = min(100.0, 5000.0 / max(dist, 50)) * confidence
                self._local_costmap[ly, lx] = max(self._local_costmap[ly, lx], cost)

        # ---- 将追踪障碍物追加到 DWA 碰撞检测列表（去重，利用第一遍已收集的数据） ----
        # 雷达暂时扫不到已固化的障碍物时，DWA 仍能"记住"它们的位置
        tracked_added = set()
        for local_x, local_y, dist, key in _tracked_for_dwa:
            if key in current_keys or key in tracked_added:
                continue
            tracked_added.add(key)
            self._local_obstacles.append((local_x, local_y, dist))

        # ---- 注入静态地图墙壁：雷达漏掉的墙壁由地图兜底 ----
        static_grid = self.mapper.map
        if static_grid is not None:
            rx_m, ry_m = static_grid.world_to_map(robot_x, robot_y)
            sample_radius_cells = 100    # 100 格 = 1000mm
            stride = 3                    # 每 3 格 (30mm) 采样一次
            occ_thresh = static_grid.occ_thresh
            static_added = 0
            for dy_c in range(-sample_radius_cells, sample_radius_cells + 1, stride):
                for dx_c in range(-sample_radius_cells, sample_radius_cells + 1, stride):
                    mx = rx_m + dx_c
                    my = ry_m + dy_c
                    if 0 <= mx < static_grid.size and 0 <= my < static_grid.size:
                        if static_grid.log_odds[my, mx] > occ_thresh:
                            wx, wy = static_grid.map_to_world(mx, my)
                            # 去重：如果 50mm 内已有雷达点，跳过
                            key = (round(wx / 50.0), round(wy / 50.0))
                            if key in current_keys:
                                continue
                            dx_w = wx - robot_x
                            dy_w = wy - robot_y
                            dist_w = math.hypot(dx_w, dy_w)
                            if dist_w < 30 or dist_w > 1100:
                                continue
                            local_x = dx_w * math.cos(robot_theta) + dy_w * math.sin(robot_theta)
                            local_y = -dx_w * math.sin(robot_theta) + dy_w * math.cos(robot_theta)
                            self._local_obstacles.append((local_x, local_y, dist_w))
                            static_added += 1
            if static_added > 50:
                print(f"[DWA] 静态地图注入 {static_added} 个墙壁点（雷达盲区兜底）")
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
    # 标准 DWA（动态窗口法）局部规划
    # 在速度空间 (vx, vy, vw) 中采样，推演轨迹，评估多目标代价后选择最优
    # ============================================================
    def _dynamic_window_avoidance(self, x, y, theta, base_vx, base_vy, base_vw, emergency=False) -> Tuple[float, float, float]:
        """DWA: 两级碰撞检测 + 路径跟踪选择

        硬边界 (125mm): 物理车身 → 碰撞 = 绝对禁止
        舒适边界 (300mm): 车身+边距 → 碰撞 = 标记为"紧贴"，仅在没有舒适候选时允许
        """
        self._update_local_costmap(x, y, theta)

        # ---- 障碍物世界坐标 ----
        cos0, sin0 = math.cos(theta), math.sin(theta)
        obs_world = [(x + ox * cos0 - oy * sin0,
                       y + ox * sin0 + oy * cos0)
                      for ox, oy, _ in self._local_obstacles]

        # ---- 两级碰撞半宽 ----
        # 硬边界：物理车身，不可逾越
        PHYS_MARGIN = 0.0
        phys_hl = self.robot_length_mm / 2.0 + PHYS_MARGIN   # 125mm
        phys_hw = self.robot_width_mm / 2.0 + PHYS_MARGIN    # 125mm

        # 舒适边界：期望保持的距离
        COMFORT_MARGIN = 175.0
        comfort_hl = phys_hl + COMFORT_MARGIN                 # 300mm
        comfort_hw = phys_hw + COMFORT_MARGIN                 # 300mm

        # ---- 速度窗口 ----
        dt = 0.3
        vx_min = max(-self.MAX_VX, base_vx - 430.0 * dt)
        vx_max = min(self.MAX_VX, base_vx + 430.0 * dt)
        vy_min = max(-self.MAX_VY, base_vy - 430.0 * dt)
        vy_max = min(self.MAX_VY, base_vy + 430.0 * dt)
        vw_min = max(-self.MAX_VW, base_vw - 2.0 * dt)
        vw_max = min(self.MAX_VW, base_vw + 2.0 * dt)

        predict_time = 1.5    # 预测 1.5 秒，覆盖 225mm（150mm/s），确保前方障碍物可见
        time_step = 0.15      # 10 步推演

        # ---- Lookahead 目标点 ----
        if self.waypoints and self.current_wp < len(self.waypoints):
            target = self._find_lookahead_point(x, y, self._adaptive_lookahead())
        elif self.waypoints:
            target = self.waypoints[-1]
        else:
            target = None

        # ---- 路径导向采样：密集采样在 A* 方向附近，稀疏覆盖绕行方向 ----
        if target:
            # 目标在机器人坐标系中的方向
            dx_w = target[0] - x
            dy_w = target[1] - y
            dx_local = dx_w * cos0 + dy_w * sin0
            dy_local = -dx_w * sin0 + dy_w * cos0

            # 理想速度：朝向目标点
            ideal_vx = np.clip(self.KP_V * dx_local, vx_min, vx_max)
            ideal_vy = np.clip(self.KP_V * dy_local * 0.7, vy_min, vy_max)
            target_angle = math.atan2(dy_local, max(dx_local, 1.0))
            ideal_vw = np.clip(self.KP_W * target_angle, vw_min, vw_max)

            # 在理想速度附近密集采样（5个点），边缘稀疏覆盖（各1个点）
            # vx: [vx_min] [3点近ideal] [vx_max]
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

        # ---- 评估所有候选轨迹 ----
        comfort_safe = []   # 舒适边界内无障碍
        tight_ok = []       # 硬边界 OK 但舒适边界有侵入
        all_cands = []

        for cvx in vx_samples:
            for cvy in vy_samples:
                for cvw in vw_samples:
                    px, py, ptheta = x, y, theta
                    hard_collision = False
                    comfort_collision = False
                    min_cl = float('inf')
                    steps = int(predict_time / time_step)

                    # ---- t=0 碰撞检测：检查当前位置是否已在障碍物内 ----
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

                            # 舒适边界距离（用于评分）
                            d_comfort = max(abs(dx_l) - comfort_hl, abs(dy_l) - comfort_hw)
                            min_cl = min(min_cl, d_comfort)

                            # 硬碰撞检测：物理车身
                            if abs(dx_l) < phys_hl and abs(dy_l) < phys_hw:
                                hard_collision = True
                                break
                            # 舒适碰撞检测：舒适边界
                            if abs(dx_l) < comfort_hl and abs(dy_l) < comfort_hw:
                                comfort_collision = True
                        if hard_collision:
                            break

                    path_dist = math.hypot(target[0] - px, target[1] - py) if target else 0.0

                    cand = {'vx': cvx, 'vy': cvy, 'vw': cvw,
                            'hard_collision': hard_collision,
                            'comfort_collision': comfort_collision,
                            'clearance': min_cl, 'path_dist': path_dist}
                    all_cands.append(cand)

                    if not hard_collision:
                        tight_ok.append(cand)
                        if not comfort_collision:
                            comfort_safe.append(cand)

        # ---- 选择最优轨迹 ----
        if comfort_safe:
            # 开阔空间：综合评分 = 路径距离(60%) + 远离障碍物(40%)
            # 有多个安全轨迹时，主动选择离障碍物更远的，充分利用可用空间
            cl_vals = [c['clearance'] for c in comfort_safe]
            pd_vals = [c['path_dist'] for c in comfort_safe]
            cl_min_v, cl_max_v = min(cl_vals), max(cl_vals)
            pd_min_v, pd_max_v = min(pd_vals), max(pd_vals)
            cl_r = cl_max_v - cl_min_v + 1e-6
            pd_r = pd_max_v - pd_min_v + 1e-6
            for c in comfort_safe:
                cl_norm = (c['clearance'] - cl_min_v) / cl_r       # 0(最贴边) ~ 1(最开阔)
                pd_norm = 1.0 - (c['path_dist'] - pd_min_v) / pd_r  # 0(最远) ~ 1(最近)
                c['score'] = pd_norm * 0.6 + cl_norm * 0.4  # 60%跟路径 + 40%远离障碍
            best = max(comfort_safe, key=lambda c: c['score'])
        elif tight_ok:
            # 紧贴模式 → 综合评分：clearance(60%) + path_dist(40%)
            # 纯两段筛选会让 path_dist 压制 clearance → 不绕行
            cl_vals = [c['clearance'] for c in tight_ok]
            pd_vals = [c['path_dist'] for c in tight_ok]
            cl_min, cl_max = min(cl_vals), max(cl_vals)
            pd_min, pd_max = min(pd_vals), max(pd_vals)
            cl_r = cl_max - cl_min + 1e-6
            pd_r = pd_max - pd_min + 1e-6
            for c in tight_ok:
                c['score'] = (c['clearance'] - cl_min) / cl_r * 0.6 \
                           + (1.0 - (c['path_dist'] - pd_min) / pd_r) * 0.4
            best = max(tight_ok, key=lambda c: c['score'])
        elif all_cands:
            # 全部碰撞 → 选 clearance 最好的
            best = max(all_cands, key=lambda c: c['clearance'])
        else:
            return base_vx, base_vy, base_vw

        vx, vy, vw = best['vx'], best['vy'], best['vw']

        # ---- 紧贴模式降速：按 clearance 比例，越靠近硬边界越慢 ----
        if best.get('comfort_collision', False):
            # clearance ∈ [-COMFORT_MARGIN, 0] (舒适边界→硬边界)
            # 映射到 speed ∈ [0.5, 1.0]，窄道中更快通过（原 0.3~0.8）
            cl = max(-COMFORT_MARGIN, min(0.0, best['clearance']))
            speed_scale = 0.5 + 0.5 * (cl + COMFORT_MARGIN) / COMFORT_MARGIN
            vx *= speed_scale
            vy *= speed_scale

        # ---- 限幅 ----
        vx = max(-self.MAX_VX, min(self.MAX_VX, vx))
        vy = max(-self.MAX_VY, min(self.MAX_VY, vy))
        vw = max(-self.MAX_VW, min(self.MAX_VW, vw))

        # ---- 低通滤波平滑 + 输出变化率限制（防突然加速）----
        if not hasattr(self, '_last_dwa_vx'):
            self._last_dwa_vx = base_vx
            self._last_dwa_vy = base_vy
            self._last_dwa_vw = base_vw
        alpha = 0.3  # 更强的平滑（原 0.5→0.3）
        vx_smooth = alpha * vx + (1 - alpha) * self._last_dwa_vx
        vy_smooth = alpha * vy + (1 - alpha) * self._last_dwa_vy
        vw_smooth = alpha * vw + (1 - alpha) * self._last_dwa_vw

        # 输出变化率限制：单周期最大变化量（防突然加速/急转）
        max_dv = 80.0    # mm/s² 等效 (80mm/s / 0.1s 控制周期)
        max_dw = 0.25    # rad/s²
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
    # 导航主循环（已移除 _update_map_with_pointcloud 调用）
    # ============================================================
    def update(self, x: float, y: float, theta: float) -> Tuple[float, float, float]:
        if self.state in ("IDLE", "FAILED", "DONE"):
            return 0.0, 0.0, 0.0

        # ============================================================
        # 起点保护 — 最高优先级！
        # 每次导航启动时，先在起点原地旋转，让车头大致对准路径方向，
        # 然后再开始路径跟踪。平移和旋转分离，避免边转边移导致位姿偏移。
        # 不管任何其他变量，起点保护必须执行完成。
        # ============================================================
        if self._startup_align_phase < 2:
            return self._startup_alignment_step(x, y, theta)

        if self.state == "ALIGNING":
            return self._align_heading_step(x, y, theta)

        if self._reached_goal(x, y):
            self._nav_count += 1  # 到达目标才算一次完整导航
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
                    if self.waypoints:
                        target = self.waypoints[-1]
                        if self.set_target(target[0], target[1], self.target_theta):
                            self._total_replan_count += 1
                            return 0.0, 0.0, 0.0
                        else:
                            # 重规划失败不终止导航，退化为紧急避障续命
                            print("[NAV] 卡住重规划失败，改用紧急避障")
                            return 0.0, 0.0, 0.0
                while self._stuck_pos_history and now - self._stuck_pos_history[0][2] > self._stuck_check_duration:
                    self._stuck_pos_history.popleft()

        # 瞬时卡住检测：不动超过2秒触发重规划，不后退
        if self.state not in ("ALIGNING", "EVADE_EMERGENCY", "AVOIDING") and self.last_pos:
            moved = math.hypot(x - self.last_pos[0], y - self.last_pos[1])
            if moved > 30:
                self.stuck_timer = now
            elif now - self.stuck_timer > 2.0 and now > self.stuck_recovery_until:
                print("[NAV] 瞬时卡住检测触发！触发重规划")
                self.stuck_recovery_until = now + 2.0  # 2秒内不重复触发
                self.stuck_timer = now
                if self.waypoints:
                    target = self.waypoints[-1]
                    self.set_target(target[0], target[1], self.target_theta)
                return 0.0, 0.0, 0.0
        self.last_pos = (x, y)

        self._sync_waypoint_index(x, y)

        # 横向偏差过大 -> 重规划
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

        # 障碍物附近重置卡住计时器：靠近障碍物时移动慢是正常的，不要触发后退振荡
        if front_dist < 400.0 or obstacle_level != "CLEAR":
            self.stuck_timer = now

        self.state = "FOLLOWING"
        base_vx, base_vy, base_vw = self._pure_pursuit_step(x, y, theta)

        # 前方障碍物距离越近减速越强：250mm 开始减速，80mm 降到最低
        if front_dist < 250.0:
            speed_scale = max(0.15, (front_dist - 80.0) / 170.0)
            base_vx = base_vx * speed_scale
            base_vy = base_vy * speed_scale
            if front_dist < 150.0:
                print(f"[BRAKE] 前方障碍物 {front_dist:.0f}mm，速度缩放至 {speed_scale:.2f}")

        # 终点精确到位：距离目标 < 200mm 时跳过 DWA，纯追踪直达
        dist_to_goal = math.hypot(self.waypoints[-1][0] - x, self.waypoints[-1][1] - y) \
                       if self.waypoints else float('inf')

        # DWA 避障：安全过滤 + 路径跟踪选择
        if self._dw_enabled and dist_to_goal >= 200.0:
            if obstacle_level == "EMERGENCY":
                # 紧急：跳过 DWA 采样，直接搜索安全逃离方向
                vx, vy, vw = self._evade_obstacle(x, y, theta)
                self.state = "EVADE_EMERGENCY"
            else:
                vx, vy, vw = self._dynamic_window_avoidance(
                    x, y, theta, base_vx, base_vy, base_vw)
                if obstacle_level == "CAUTION":
                    self.state = "AVOIDING"
        else:
            vx, vy, vw = base_vx, base_vy, base_vw

        # 导航起步速度渐变：前2秒内逐步从30%加速到100%
        if getattr(self, '_nav_start_time', 0) > 0:
            elapsed = now - self._nav_start_time
            if elapsed < 2.0:
                ramp = 0.3 + 0.7 * (elapsed / 2.0)
                vx *= ramp
                vy *= ramp
                vw *= ramp

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
        """检查前方路径点是否被障碍物阻挡（用当前地图+实时点云+追踪障碍物，不用冻结快照）。
        返回 True 表示路径被阻挡，需要重规划。"""
        if not self.waypoints or self.current_wp >= len(self.waypoints):
            return False
        grid = self.mapper.map
        # 用当前实时地图 + 实时点云判断
        static_occ = grid.log_odds > grid.occ_thresh
        _, world_pts = self.mapper.get_latest_points()
        # 快速构建实时点云占用掩码（只检查路径点附近）
        pc_occ = set()
        if world_pts:
            for wx, wy in world_pts:
                mx, my = grid.world_to_map(wx, wy)
                if 0 <= mx < grid.size and 0 <= my < grid.size:
                    pc_occ.add((mx, my))

        # 追踪中的障碍物栅格（含未固化的）
        tracked_occ = set()
        for obs_data in self._stable_obstacles.values():
            for mx, my in obs_data['cells']:
                if 0 <= mx < grid.size and 0 <= my < grid.size:
                    tracked_occ.add((mx, my))

        # 检查接下来的 3 个路径点（余量由 channel_margin_mm 决定）
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
            if dist > 2000 or dist < 20:  # 只过滤 20mm 内自身反射
                continue
            angle = self._normalize_angle(math.atan2(dy, dx) - theta)
            # 扩大到 ±70°，确保能检测到正前方障碍物
            if abs(angle) < math.radians(70):
                front_min = min(front_min, dist)
        return front_min

    def set_target(self, x: float, y: float, theta: float = None) -> bool:
        self.target_theta = theta

        start = (self.mapper.pose.x, self.mapper.pose.y)
        goal = (x, y)

        was_frozen = self._plan_frozen  # 记录之前的冻结状态
        self._freeze_planning_map()

        raw_path = self._astar(start, goal)
        if not raw_path:
            if not was_frozen:
                # 首次规划失败 → 真正的不可达
                self.state = "FAILED"
                self._unfreeze_planning_map()
                print(f"[NAV] A* 首次规划失败，目标不可达")
            else:
                # 重规划失败 → 保留旧路径继续走
                print(f"[NAV] A* 重规划失败，保留旧路径继续")
            return False

        self.path = raw_path
        inflated = self._get_planning_inflated_grid()
        self.waypoints = self._smooth_path(raw_path, inflated)
        self.current_wp = 0

        self.state = "FOLLOWING"
        self.last_replan_time = time.time()
        self.stuck_timer = time.time()
        self.last_pos = (self.mapper.pose.x, self.mapper.pose.y)

        # 导航起步速度渐变：记录起动时刻，前2秒内逐步放开限速
        self._nav_start_time = time.time()

        self._stuck_check_start = time.time()
        self._stuck_pos_history.clear()

        self._align_settle_until = 0.0
        self._alignment_ack_sent = False
        self._send_alignment_ack = False
        self._align_stable_count = 0

        # 全向底盘：跳过启动对齐，直接进入路径跟踪
        self._startup_align_phase = 2
        self._startup_align_stable = 0

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
        self._unfreeze_planning_map()
        print("[NAV] 导航已取消，规划地图已解冻")

    # ============================================================
    # 起点保护 — 最高优先级！
    # 首次导航：向右上方移动，直到离开起点区域（>200mm）。
    # 后续导航：原地旋转，让车头大致对准路径方向后再开始跟踪。
    # 保护期间 _startup_align_phase < 2，update() 直接 return，
    # 任何其他逻辑（卡住检测、DWA、pure pursuit）都不会执行。
    # ============================================================
    def _startup_alignment_step(self, x: float, y: float, theta: float) -> Tuple[float, float, float]:
        # ---- 首次导航：起点保护，向右上方移动，直到离开起点区域 ----
        if self._nav_count == 0:
            # 记录起始位置（第一帧）
            if not hasattr(self, '_startup_start_pos'):
                self._startup_start_pos = (x, y)

            frame = self._startup_align_stable
            self._startup_align_stable += 1

            # 已移动距离
            moved = math.hypot(x - self._startup_start_pos[0],
                               y - self._startup_start_pos[1])

            # 完成条件：至少 3 帧 且 移动超过 150mm 才放行
            if frame >= 3 and moved > 150.0:
                self._startup_align_phase = 2
                delattr(self, '_startup_start_pos')
                print(f"[STARTUP] 起点保护完成 (移动 {moved:.0f}mm) → 开始路径跟踪")
                return 0.0, 0.0, 0.0
            return 100.0, -100.0, 0.0

        # ---- 后续导航：原地旋转对准路径方向 ----
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

        if abs_diff >= math.radians(10.0):
            self._align_settle_until = 0.0
            self._alignment_ack_sent = False
            self._align_stable_count = 0

            # 对准阶段使用独立的低 KP，避免超调振荡
            align_KP = 0.7

            # 角度差 < 20° 时渐进减速，防止冲过头
            if abs_diff < math.radians(20.0):
                scale = abs_diff / math.radians(20.0)  # 0→1，角度越小越慢
                vw_cap = self.MAX_VW * 0.25 + self.MAX_VW * 0.75 * scale  # 25%~100% MAX_VW
            else:
                vw_cap = self.MAX_VW

            vw_raw = align_KP * angle_diff
            vw = max(-vw_cap, min(vw_cap, vw_raw))
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
        """获取全局路径在点(x,y)最近处的切线方向（弧度）。
        用于 DWA 评估轨迹终点是否与全局路径方向一致，
        防止机器人左右徘徊而忽略全局路径走向。"""
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

        # DWA 绕行后机器人可能离当前路径点很远，扫描前方找最近的点
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

        # 狭窄通道检测：两侧都有障碍物时自动降速（阈值 400mm）
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
                # 扩大检测角度到 10°~90°，让车头能感知正前方两侧的障碍物
                if 10 <= a_deg < 90:
                    left_wall = min(left_wall, d)
                elif -90 < a_deg <= -10:
                    right_wall = min(right_wall, d)
            if left_wall < float('inf') and right_wall < float('inf'):
                channel_width = left_wall + right_wall
                if channel_width < 400:
                    channel_scale = max(0.4, channel_width / 400.0)
                    max_vx *= channel_scale
                    max_vy *= channel_scale
                    if channel_width < 350:
                        print(f"[CHANNEL] 狭窄通道 {channel_width:.0f}mm，降速至 {channel_scale:.1f}")

        # 全向底盘：目标在后方时 local_x 为负，自然产生后退速度
        vx_raw = kp_v * local_x

        vy_raw = self.KP_V * local_y * 0.7  # 全向底盘侧移与前移同等有效

        v_mag = math.hypot(vx_raw, vy_raw)
        if v_mag > max_vx:
            ratio = max_vx / v_mag
            vx_raw *= ratio
            vy_raw *= ratio

        vx = max(-max_vx, min(max_vx, vx_raw))  # 全向底盘允许后退
        vy = max(-max_vy, min(max_vy, vy_raw))

        # 路径回归约束（极轻量，DWA 避障后有自然回归趋势即可）
        path_lateral_err, path_proj = self._calc_path_projection(x, y)
        regress_gain = 0.3 if channel_width < 450 else 0.5
        if abs(path_lateral_err) > 10.0 and len(self.waypoints) > 2:
            reg_dx = path_proj[0] - x
            reg_dy = path_proj[1] - y
            reg_dist = math.hypot(reg_dx, reg_dy)
            if reg_dist > 5:
                vy_correction = -path_lateral_err * regress_gain
                max_vy_correction = self.MAX_VY * 0.15 if channel_width < 450 else self.MAX_VY * 0.25
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

        # 大角度差：先原地转向对准，再前进（避免边转边前进的螺旋轨迹）
        if abs_angle > math.radians(90):
            vx, vy = 0.0, 0.0       # 原地转向，不平移
            vw = max(-0.5, min(0.5, self.KP_W * vw_err * 0.5))
            print(f"[TURN] 角度差 {math.degrees(abs_angle):.1f}°，原地转向对准目标")
        elif abs_angle > math.radians(45):
            vx *= 0.2                # 微进 + 主转向
            vy *= 0.2
            vw *= 0.6
        elif abs_angle > math.radians(20):
            vw *= 0.8                # 轻微调整，不减速

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

            # 360° 全向检测：-180° 到 +180°
            if abs(angle) < math.radians(45):
                front_min = min(front_min, dist)
            elif math.radians(45) <= angle < math.radians(135):
                left_min = min(left_min, dist)
            elif math.radians(-135) < angle <= math.radians(-45):
                right_min = min(right_min, dist)
            else:
                # ±135° ~ ±180° 为正后方
                rear_min = min(rear_min, dist)

        # 四周统一：120mm 紧急（立即避障），250mm 内进入 CAUTION 准备避让
        # ★ 侧面和后方也检查 EMERGENCY，防止机器人平行于墙壁时蹭墙
        if front_min < 120 or left_min < 120 or right_min < 120:
            return "EMERGENCY"
        if front_min < 250 or left_min < 250 or right_min < 250 or rear_min < 250:
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

        # 前方安全（>300mm）：退出避障，交给 DWA/pure pursuit 正常导航
        # 侧面有障碍物是正常的（刚绕过还在旁边），不应继续横移
        if front_min > 300:
            print(f"[EVADE] 前方已安全({front_min:.0f}mm)，退出避障→正常导航")
            base_vx, base_vy, base_vw = self._pure_pursuit_step(x, y, theta)
            return base_vx * 0.5, base_vy * 0.5, base_vw * 0.5


        # 基于路径方向搜索最优安全方向
        best_dir = None
        best_score = -float('inf')

        # 优先测试路径方向及附近方向
        # 当已经非常靠近障碍物时，优先搜索侧向方向（左右横移）而不是正前方
        test_dirs = []
        if front_min < 200:
            # 极近时优先侧向：先路径侧向，再斜向，最后正前
            for priority_offset in [90, -90, 75, -75, 60, -60, 45, -45, 30, -30, 15, -15, 0, 120, -120]:
                test_dirs.append(self._normalize_angle(path_dir + math.radians(priority_offset)))
        elif right_min < 200:
            # 右侧有障碍物：优先向左（正角度偏移）搜索
            for priority_offset in [90, 75, 60, 45, 30, 15, 0, -15, -30, -45, -60, -75, -90, 120, -120]:
                test_dirs.append(self._normalize_angle(path_dir + math.radians(priority_offset)))
        elif left_min < 200:
            # 左侧有障碍物：优先向右（负角度偏移）搜索
            for priority_offset in [-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90, 120, -120]:
                test_dirs.append(self._normalize_angle(path_dir + math.radians(priority_offset)))
        else:
            for priority_offset in [0, 15, -15, 30, -30, 45, -45, 60, -60, 75, -75, 90, -90, 120, -120]:
                test_dirs.append(self._normalize_angle(path_dir + math.radians(priority_offset)))

        # 检查前方多个距离点，近距离即可，不要要求700mm都clear（窄通道做不到）
        check_dists = [100, 200, 350]
        # 安全半径：使用矩形对角线半长 + 安全余量，确保角落也不会蹭到
        half_diag_phys = math.hypot(self.robot_length_mm / 2.0, self.robot_width_mm / 2.0)  # ≈177mm
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
            # 侧面有障碍物时，奖励远离障碍物方向
            if right_min < 350:
                # 右侧障碍物：奖励向左(正 vy)的方向
                if math.sin(test_rad) > 0:
                    score += (350.0 - right_min) * 0.3
            if left_min < 350:
                # 左侧障碍物：奖励向右(负 vy)的方向
                if math.sin(test_rad) < 0:
                    score += (350.0 - left_min) * 0.3

            if score > best_score:
                best_score = score
                best_dir = test_rad

        if best_dir is not None:
            # 根据前方距离决定速度，窄道中侧向滑出要更快才有效果
            if front_min < 300:
                speed = 80.0
            elif front_min < 500:
                speed = 150.0
            else:
                speed = 120.0

            vx = speed * math.cos(best_dir)
            vy = speed * math.sin(best_dir)
            # 全向底盘允许后退，不再强制截断负vx
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
        cost_grid = self._get_cost_grid()  # 距离变换代价网格
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

        # 不再大面积清空起点周围，避免A*路径贴着被抹掉的墙壁走。
        # _get_planning_inflated_grid 已保证起点中心可通行，足够A*起步。

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

                # 距离变换代价：离障碍物越近，代价越高（指数衰减，梯度陡峭）
                # 指数衰减：gain * exp(-dist/decay)，近处代价极高，远处迅速趋近于0
                if cost_grid is not None:
                    dist_cells = cost_grid[ny, nx]
                    occ_penalty = self._obstacle_cost_gain * math.exp(-dist_cells / self._obstacle_cost_decay)
                else:
                    occ_penalty = 0.0

                # 静态墙壁额外代价：指数衰减，强力推开路径到走廊中心
                if self._static_cost_grid is not None:
                    static_dist = self._static_cost_grid[ny, nx]
                    static_penalty = self._static_cost_gain * math.exp(-static_dist / self._static_cost_decay)
                else:
                    static_penalty = 0.0

                tentative = g_score[cy, cx] + cost + occ_penalty + static_penalty
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

        # 获取膨胀地图用于碰撞检查（平滑后路径不能侵入膨胀区域）
        if inflated_grid is None:
            inflated_grid = self._get_planning_inflated_grid()
        grid = self.mapper.map

        sampled = [path[0]]
        for p in path[1:-1]:
            if math.hypot(p[0] - sampled[-1][0], p[1] - sampled[-1][1]) > 80:
                sampled.append(p)
        sampled.append(path[-1])

        smooth = [list(p) for p in sampled]
        # 降低平滑权重，避免过度切角导致路径缩进障碍物
        for _ in range(10):
            for i in range(1, len(smooth) - 1):
                ax = (smooth[i-1][0] + smooth[i+1][0]) * 0.5
                ay = (smooth[i-1][1] + smooth[i+1][1]) * 0.5
                smooth[i][0] = 0.8 * smooth[i][0] + 0.2 * ax
                smooth[i][1] = 0.8 * smooth[i][1] + 0.2 * ay

        # 关键修复：平滑后检查每个点是否在膨胀栅格上，如果是则回退到原始路径点
        restored = []
        for i, p in enumerate(smooth):
            mx, my = grid.world_to_map(p[0], p[1])
            if 0 <= mx < grid.size and 0 <= my < grid.size:
                if inflated_grid[my, mx]:
                    # 平滑后这个点撞膨胀区了，回退到原始采样点
                    orig = sampled[i]
                    restored.append((orig[0], orig[1]))
                    print(f"[SMOOTH] 路径点 {i} 平滑后侵入膨胀区，已回退到原始点")
                    continue
            restored.append((p[0], p[1]))

        return restored

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