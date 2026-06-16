"""占用栅格地图 — 清晰边界，无黑点堆积"""

import math
import numpy as np


def binary_erosion(binary_array, iterations=1):
    result = binary_array.copy().astype(np.uint8)
    for _ in range(iterations):
        eroded = np.zeros_like(result)
        eroded[1:-1, 1:-1] = (
            result[0:-2, 1:-1] &
            result[2:, 1:-1] &
            result[1:-1, 0:-2] &
            result[1:-1, 2:]
        )
        result = eroded
    return result.astype(bool)


def binary_dilation(binary_array, iterations=1):
    result = binary_array.copy().astype(np.uint8)
    for _ in range(iterations):
        dilated = result.copy()
        dilated[0:-1, :] |= result[1:, :]
        dilated[1:, :] |= result[0:-1, :]
        dilated[:, 0:-1] |= result[:, 1:]
        dilated[:, 1:] |= result[:, 0:-1]
        result = dilated
    return result.astype(bool)


class OccupancyGridMap:
    """
    占用栅格地图 — 清晰边界版
    """

    def __init__(self, size_mm: int = 8000, resolution_mm: int = 20):
        self.size_mm = size_mm
        self.resolution = resolution_mm
        self.size = size_mm // resolution_mm
        self.log_odds = np.zeros((self.size, self.size), dtype=np.float32)

        # 强占用：扫到就黑
        self.log_occ = 6.0
        # 强空闲：没扫到就白（快速清除误标记）
        self.log_free = -1.0
        self.max_log = 30.0
        self.min_log = -10.0
        # 占用阈值：几乎一次命中就黑
        self.occ_thresh = 4.0
        # 空闲阈值：快速变白
        self.free_thresh = -2.0

        self.endpoint_margin = 2
        self.offset = self.size // 2

        # === 静态地图缓存 ===
        self._static_display = None

    def world_to_map(self, x: float, y: float):
        mx = int(-y / self.resolution + self.offset)
        my = int(-x / self.resolution + self.offset)
        return mx, my

    def map_to_world(self, mx: int, my: int):
        y = -(mx - self.offset) * self.resolution
        x = -(my - self.offset) * self.resolution
        return x, y

    def _bresenham_line(self, x0, y0, x1, y1):
        """标准 Bresenham：射线经过=空闲，终点=占用"""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            in_bounds = (0 <= x0 < self.size and 0 <= y0 < self.size)
            if not in_bounds:
                if x0 == x1 and y0 == y1:
                    break
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x0 += sx
                if e2 < dx:
                    err += dx
                    y0 += sy
                continue

            is_endpoint = (x0 == x1 and y0 == y1)

            if is_endpoint:
                self.log_odds[y0, x0] += self.log_occ
            else:
                self.log_odds[y0, x0] += self.log_free

            self.log_odds[y0, x0] = np.clip(
                self.log_odds[y0, x0], self.min_log, self.max_log
            )

            if x0 == x1 and y0 == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def update_ray(self, robot_x: float, robot_y: float,
                   point_x: float, point_y: float):
        dist = math.hypot(point_x - robot_x, point_y - robot_y)
        if dist < 100:
            return
        if dist > 5000:
            return

        half = self.size_mm / 2

        if not (-half <= point_x <= half and -half <= point_y <= half):
            dx = point_x - robot_x
            dy = point_y - robot_y
            d = math.hypot(dx, dy)
            if d > 0:
                scale = min(
                    abs((half - abs(robot_x)) / dx) if dx != 0 else float('inf'),
                    abs((half - abs(robot_y)) / dy) if dy != 0 else float('inf'),
                    1.0
                )
                point_x = robot_x + dx * scale
                point_y = robot_y + dy * scale

        rx, ry = self.world_to_map(robot_x, robot_y)
        px, py = self.world_to_map(point_x, point_y)

        if 0 <= rx < self.size and 0 <= ry < self.size:
            self._bresenham_line(rx, ry, px, py)

    def get_display(self, sharpen: bool = True):
        """生成显示图像：黑=占用，白=空闲，灰=未知"""

        # 静态地图模式下直接返回缓存
        if self._static_display is not None:
            return self._static_display

        h, w = self.size, self.size
        display = np.full((h, w), 128, dtype=np.uint8)

        # 严格分类：强占用=黑，强空闲=白，中间=灰
        occupied = self.log_odds > self.occ_thresh
        free = self.log_odds < self.free_thresh

        display[occupied] = 0
        display[free] = 255

        if sharpen:
            occ_mask = display == 0

            # 面积过滤：去除孤立小黑点（噪点）
            try:
                from scipy import ndimage
                labeled, num_features = ndimage.label(occ_mask)
                sizes = ndimage.sum(occ_mask, labeled, range(1, num_features + 1))
                keep_mask = sizes >= 7
                occ_mask_clean = np.zeros_like(occ_mask)
                for i, keep in enumerate(keep_mask, start=1):
                    if keep:
                        occ_mask_clean[labeled == i] = True

                occ_mask_clean = binary_dilation(occ_mask_clean, iterations=1)
                occ_mask_clean = binary_erosion(occ_mask_clean, iterations=1)

                display = np.full((h, w), 128, dtype=np.uint8)
                display[occ_mask_clean] = 0
                display[free] = 255
            except ImportError:
                pass  # 没有 scipy 就不做 sharpen

        return display

    def save(self, filepath: str):
        import matplotlib.pyplot as plt
        plt.imsave(filepath, self.get_display(), cmap='gray')

    def load_from_image(self, filepath: str):
        """从 PNG/JPG 加载已保存的栅格地图（黑=占用，白=空闲，灰=未知）"""
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("加载地图需要 Pillow，请运行: pip install Pillow")

        img = Image.open(filepath)
        if img.mode != 'L':
            img = img.convert('L')  # 强制灰度

        # 尺寸对齐（最近邻保持硬边界）
        if img.size != (self.size, self.size):
            img = img.resize((self.size, self.size), Image.NEAREST)

        arr = np.array(img, dtype=np.uint8)

        # 直接生成 display 数组并缓存，绕过 log_odds 机制
        display = np.full((self.size, self.size), 128, dtype=np.uint8)

        # 黑 (< 60) → 占用 (0)
        occupied = arr < 60
        # 白 (> 200) → 空闲 (255)
        free = arr > 200
        # 中间灰 → 未知 (128)

        display[occupied] = 0
        display[free] = 255

        # 同时设置 log_odds 供 A* 使用
        self.log_odds.fill(0.0)
        self.log_odds[occupied] = self.occ_thresh + 2.0
        self.log_odds[free] = self.free_thresh - 2.0

        # 缓存静态显示图像
        self._static_display = display

        occ = np.count_nonzero(occupied)
        fr = np.count_nonzero(free)
        print(f"[MAP] 已加载 {filepath}: {self.size}x{self.size}, "
              f"占用={occ}, 空闲={fr}, 未知={self.size*self.size - occ - fr}")

    def reset_to_loaded(self, filepath: str):
        """清空后重新加载原始地图"""
        self._static_display = None
        self.load_from_image(filepath)

    def __repr__(self):
        return f"OccupancyGridMap({self.size}x{self.size}, {self.resolution}mm/格)"