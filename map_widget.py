"""PyQt5 地图显示控件 — 栅格地图为底图（坐标对齐），叠加实时点云/轨迹/导航路径，并显示永久障碍物"""
import math
import numpy as np
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, pyqtSignal, QRectF
from PyQt5.QtGui import QPainter, QColor, QPen, QFont, QImage, QBrush, QPixmap
from mapper import LidarMapper

class MapWidget(QWidget):
    # 信号：用户在地图上点击了一个世界坐标点
    map_clicked = pyqtSignal(float, float)

    def __init__(self, mapper: LidarMapper, navigator=None, mode="combined"):
        super().__init__()
        self.mapper = mapper
        self.navigator = navigator
        self.mode = mode
        self.setMinimumSize(400, 400)
        self.scale = 0.20 # 缩放因子
        # 坐标轴偏移（水平居中、垂直居中）
        self.axis_offset_x = -0.05
        self.axis_offset_y = 0.52
        # 图像缓冲区（防止 GC 回收）
        self._grid_image_buffer = None
        self._inflated_image_buffer = None

        # ===== 叠加框：二维矩形（世界坐标 mm，可自行调整）=====
        # 参数格式: QRectF(中心点X, 中心点Y, 宽度W, 高度H)
        self.overlay_green_rect = QRectF(1590, 10, 350, 300)     # 绿框：中心(0,0)，宽高1000mm
        self.overlay_yellow_rect = QRectF(20, -1580, 420, 350) # 黄框：中心
        self.overlay_brown_rect = QRectF(800 ,-800, 2000, 2000)       # 棕色镂空框（只描边不填充）

        # 文字标签的世界坐标位置 (可自行调整)
        self.green_label_pos = (1630, 500)    # "好果存放区" 位置 (世界坐标 mm)
        self.yellow_label_pos = (-70, -1910)  # "坏果存放区" 位置 (世界坐标 mm)

        # WebSocket 截图裁剪区域（涵盖好果存放区 + 坏果存放区）
        self.crop_rect = QRectF(780, -780, 2400, 3000)

        # ===== 叠加图片（世界坐标 mm，可自行调整）====
        self._overlay_pixmap = QPixmap("2.jpg")
        self.overlay_img_center = (420, -430)       # 图片中心位置 (世界坐标 mm)
        self.overlay_img_size_mm = (0, 0)  # 图片显示大小 (宽, 高) mm

        self._overlay_pixmap2 = QPixmap("3.jpg")
        self.overlay_img_center2 = (1220, -1180)      # 图片中心位置 (世界坐标 mm)
        self.overlay_img_size_mm2 = (0, 0) # 图片显示大小 (宽, 高) mm

        # 启用鼠标追踪以支持点击导航
        self.setMouseTracking(True)

    def wheelEvent(self, event):
        """鼠标滚轮缩放：以鼠标位置为中心缩放地图"""
        delta = event.angleDelta().y()
        zoom_factor = 1.15 if delta > 0 else 1.0 / 1.15
        old_scale = self.scale
        new_scale = old_scale * zoom_factor
        # 限制缩放范围
        if new_scale < 0.03 or new_scale > 3.0:
            return
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            self.scale = new_scale
            self.update()
            return
        # 滚轮时鼠标所在屏幕坐标
        pos = event.position()
        mx, my = pos.x(), pos.y()
        # 缩放前鼠标指向的世界坐标
        wx, wy = self.screen_to_world(mx, my, w, h)
        # 应用新缩放
        self.scale = new_scale
        # 调整偏移量，使鼠标指向的世界坐标保持在鼠标位置不动
        self.axis_offset_x = (wy * self.scale + mx - w // 2) / w
        self.axis_offset_y = (wx * self.scale + my) / h
        self.update()

    def world_to_screen(self, wx: float, wy: float, w: int, h: int):
        cx = w // 2 + int(self.axis_offset_x * w)
        cy = int(h * self.axis_offset_y)
        sx = cx - int(wy * self.scale)
        sy = cy - int(wx * self.scale)
        return sx, sy

    def screen_to_world(self, sx: float, sy: float, w: int, h: int):
        """将屏幕坐标转换为世界坐标"""
        cx = w // 2 + int(self.axis_offset_x * w)
        cy = int(h * self.axis_offset_y)
        wy = (cx - sx) / self.scale
        wx = (cy - sy) / self.scale
        return wx, wy

    def mousePressEvent(self, event):
        """鼠标点击地图：转换为世界坐标并通过信号发出"""
        if event.button() == Qt.LeftButton:
            w, h = self.width(), self.height()
            wx, wy = self.screen_to_world(event.x(), event.y(), w, h)
            self.map_clicked.emit(wx, wy)
        super().mousePressEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        if self.mode == "combined":
            self._draw_combined(painter, w, h)
        elif self.mode == "pointcloud":
            self._draw_pointcloud(painter, w, h)
        else:
            self._draw_gridmap(painter, w, h)

        # ===== 叠加二维矩形框 =====
        # 线条宽度跟随缩放 (基准比例 0.20)
        sw = max(1, self.scale / 0.20)

        # 绿框
        gx, gy = self.overlay_green_rect.x(), self.overlay_green_rect.y()
        gw, gh = self.overlay_green_rect.width(), self.overlay_green_rect.height()
        gsx1, gsy1 = self.world_to_screen(gx - gw / 2, gy - gh / 2, w, h)
        gsx2, gsy2 = self.world_to_screen(gx + gw / 2, gy + gh / 2, w, h)
        pen = QPen(QColor(0, 255, 0, 255))
        pen.setWidth(int(3 * sw))
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(0, 255, 0, 255)))
        painter.drawRect(QRectF(min(gsx1, gsx2), min(gsy1, gsy2), abs(gsx2 - gsx1), abs(gsy2 - gsy1)))
        # 绿框标签
        font = QFont("Microsoft YaHei", int(11 * sw))
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(0, 255, 0, 255))
        glx, gly = self.world_to_screen(*self.green_label_pos, w, h)
        for i, ch in enumerate("好果存放区"):
            painter.drawText(int(glx), int(gly + i * 18 * sw), ch)

        # 黄框
        yx, yy = self.overlay_yellow_rect.x(), self.overlay_yellow_rect.y()
        yw, yh = self.overlay_yellow_rect.width(), self.overlay_yellow_rect.height()
        ysx1, ysy1 = self.world_to_screen(yx - yw / 2, yy - yh / 2, w, h)
        ysx2, ysy2 = self.world_to_screen(yx + yw / 2, yy + yh / 2, w, h)
        pen = QPen(QColor(255, 255, 0, 255))
        pen.setWidth(int(3 * sw))
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(255, 255, 0, 255)))
        painter.drawRect(QRectF(min(ysx1, ysx2), min(ysy1, ysy2), abs(ysx2 - ysx1), abs(ysy2 - ysy1)))
        # 黄框标签
        font = QFont("Microsoft YaHei", int(11 * sw))
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(255, 255, 0, 255))
        ylx, yly = self.world_to_screen(*self.yellow_label_pos, w, h)
        for i, ch in enumerate("坏果存放区"):
            painter.drawText(int(ylx), int(yly + i * 18 * sw), ch)

        # 棕色镂空框（只描边不填充）
        bx, by = self.overlay_brown_rect.x(), self.overlay_brown_rect.y()
        bw, bh = self.overlay_brown_rect.width(), self.overlay_brown_rect.height()
        bsx1, bsy1 = self.world_to_screen(bx - bw / 2, by - bh / 2, w, h)
        bsx2, bsy2 = self.world_to_screen(bx + bw / 2, by + bh / 2, w, h)
        pen = QPen(QColor(139, 69, 19, 255))
        pen.setWidth(int(25 * sw))
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(QRectF(min(bsx1, bsx2), min(bsy1, bsy2), abs(bsx2 - bsx1), abs(bsy2 - bsy1)))

        # ===== 叠加图片 =====
        if self._overlay_pixmap and not self._overlay_pixmap.isNull():
            icx, icy = self.overlay_img_center
            iw_mm, ih_mm = self.overlay_img_size_mm
            isx1, isy1 = self.world_to_screen(icx - iw_mm / 2, icy - ih_mm / 2, w, h)
            isx2, isy2 = self.world_to_screen(icx + iw_mm / 2, icy + ih_mm / 2, w, h)
            target_rect = QRectF(min(isx1, isx2), min(isy1, isy2),
                                 abs(isx2 - isx1), abs(isy2 - isy1))
            scaled = self._overlay_pixmap.scaled(int(target_rect.width()),
                                                  int(target_rect.height()),
                                                  Qt.KeepAspectRatio,
                                                  Qt.SmoothTransformation)
            painter.drawPixmap(int(target_rect.x()), int(target_rect.y()), scaled)

        # ===== 叠加图片2 =====
        if self._overlay_pixmap2 and not self._overlay_pixmap2.isNull():
            icx, icy = self.overlay_img_center2
            iw_mm, ih_mm = self.overlay_img_size_mm2
            isx1, isy1 = self.world_to_screen(icx - iw_mm / 2, icy - ih_mm / 2, w, h)
            isx2, isy2 = self.world_to_screen(icx + iw_mm / 2, icy + ih_mm / 2, w, h)
            target_rect = QRectF(min(isx1, isx2), min(isy1, isy2),
                                 abs(isx2 - isx1), abs(isy2 - isy1))
            scaled = self._overlay_pixmap2.scaled(int(target_rect.width()),
                                                  int(target_rect.height()),
                                                  Qt.KeepAspectRatio,
                                                  Qt.SmoothTransformation)
            painter.drawPixmap(int(target_rect.x()), int(target_rect.y()), scaled)

        # ===== 静态地图边框 =====
        self._draw_map_border(painter, w, h)

    def _draw_map_border(self, painter, w, h):
        """绘制静态地图的边界框"""
        grid = self.mapper.get_map_display()
        if grid is None or grid.size == 0:
            return
        h_map, w_map = grid.shape
        sw = max(1, self.scale / 0.20)
        # 地图四个角的屏幕坐标
        sx1, sy1 = self.world_to_screen(*self.mapper.map.map_to_world(0, 0), w, h)
        sx2, sy2 = self.world_to_screen(*self.mapper.map.map_to_world(w_map - 1, 0), w, h)
        sx3, sy3 = self.world_to_screen(*self.mapper.map.map_to_world(w_map - 1, h_map - 1), w, h)
        sx4, sy4 = self.world_to_screen(*self.mapper.map.map_to_world(0, h_map - 1), w, h)

        pen = QPen(QColor(255, 255, 255, 180))
        pen.setWidth(int(2 * sw))
        pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawLine(sx1, sy1, sx2, sy2)
        painter.drawLine(sx2, sy2, sx3, sy3)
        painter.drawLine(sx3, sy3, sx4, sy4)
        painter.drawLine(sx4, sy4, sx1, sy1)

    # ============================================================
    # 合并显示：栅格地图为底图 + 点云叠加
    # ============================================================
    def _draw_combined(self, painter, w, h):
        # 1. 栅格地图底图
        self._draw_gridmap_aligned(painter, w, h)
        # 2. 永久障碍物（黑色方块）
        # self._draw_permanent_obstacles(painter, w, h)
        # 2.5 临时障碍物（黑色半透明方块）
        self._draw_temporary_obstacles(painter, w, h)
        # # 3. 膨胀障碍物（半透明红色）
        self._draw_inflated_grid(painter, w, h)
        # 4. 坐标轴和网格
        self._draw_axes_and_grid(painter, w, h)
        # 5. 历史点云
        self._draw_history_points(painter, w, h)
        # 6. 最新帧点云
        self._draw_latest_points(painter, w, h)
        # 7. 机器人轨迹
        self._draw_trajectory(painter, w, h)
        # 8. 原始 A* 路径（灰色虚线）
        self._draw_raw_path(painter, w, h)
        # 9. 导航路径（平滑后）和目标点
        self._draw_navigation(painter, w, h)
        # 10. 路径点标记
        self._draw_waypoint_markers(painter, w, h)
        # 11. Lookahead 点
        self._draw_lookahead(painter, w, h)
        # 12. 安全距离圈 + 障碍物扇区（仅导航激活时显示）
        self._draw_safety_zone(painter, w, h)
        # 13. 膨胀半径圆环（始终显示）
        # self._draw_inflation_radius(painter, w, h)   # 注释掉以隐藏膨胀半径
        # 14. 机器人当前位置
        self._draw_robot(painter, w, h)
        # 15. 文字信息
        self._draw_info_text(painter, w, h)
        self._draw_nav_debug_overlay(painter, w, h)

    # ============================================================
    # 永久障碍物绘制
    # ============================================================
    def _draw_permanent_obstacles(self, painter, w, h):
        """绘制被永久固化的障碍物（黑色方块）"""
        if not self.navigator:
            return
        cells = self.navigator.get_stable_obstacle_cells()
        if not cells:
            return
        painter.setBrush(QColor(0, 0, 0, 200)) # 黑色
        painter.setPen(Qt.NoPen)
        res = self.mapper.map.resolution
        size_px = max(2, int(res * self.scale)) # 栅格在屏幕上的像素大小
        for mx, my in cells:
            wx, wy = self.mapper.map.map_to_world(mx, my)
            sx, sy = self.world_to_screen(wx, wy, w, h)
            # 绘制矩形块（以栅格中心对齐）
            painter.drawRect(sx - size_px//2, sy - size_px//2, size_px, size_px)

    def _draw_temporary_obstacles(self, painter, w, h):
        """绘制未固化的临时障碍物（黑色半透明方块）"""
        if not self.navigator:
            return
        cells = self.navigator.get_temporary_obstacle_cells()
        if not cells:
            return
        painter.setBrush(QColor(0, 0, 0, 120)) # 黑色半透明
        painter.setPen(Qt.NoPen)
        res = self.mapper.map.resolution
        size_px = max(2, int(res * self.scale))
        for mx, my in cells:
            wx, wy = self.mapper.map.map_to_world(mx, my)
            sx, sy = self.world_to_screen(wx, wy, w, h)
            painter.drawRect(sx - size_px//2, sy - size_px//2, size_px, size_px)

    # ============================================================
    # 膨胀半径圆环（始终显示）— 显示 A* 规划时实际使用的膨胀边界
    # ============================================================
    def _draw_inflation_radius(self, painter, w, h):
        """绘制膨胀半径参考圆（黄色半透明粗线），始终显示"""
        if not self.navigator:
            return
        stats = self.mapper.get_stats()
        rx, ry = stats['pose'][0], stats['pose'][1]
        sx, sy = self.world_to_screen(rx, ry, w, h)
        # 膨胀半径 = obstacle_margin * resolution（A* 栅格膨胀量）
        radius_mm = self.navigator.obstacle_margin * self.mapper.map.resolution
        r = int(radius_mm * self.scale)
        if r > 5:
            pen = QPen(QColor(255, 220, 0, 200)) # 金黄色，更明显
            pen.setWidth(3)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(sx - r, sy - r, 2 * r, 2 * r)
            # 标注文字
            font = QFont("Microsoft YaHei", 8, QFont.Bold)
            painter.setFont(font)
            painter.setPen(QColor(255, 220, 0, 220))
            painter.drawText(sx + r + 5, sy - r + 12, f"膨胀 {radius_mm:.0f}mm")
        # 最小通道宽度示意 = 车宽 + 2*channel_margin_mm（统一参数）
        min_channel_mm = self.navigator.robot_width_mm + 2 * self.navigator.channel_margin_mm if self.navigator else 350
        r_channel = int(min_channel_mm / 2.0 * self.scale)
        if r_channel > r + 5:
            pen = QPen(QColor(255, 100, 100, 100))
            pen.setWidth(1)
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(sx - r_channel, sy - r_channel, 2 * r_channel, 2 * r_channel)

    # ============================================================
    # 栅格地图底图（坐标对齐）
    # ============================================================
    def _draw_gridmap_aligned(self, painter, w, h):
        grid = self.mapper.get_map_display()
        if grid is None or grid.size == 0:
            painter.fillRect(self.rect(), QColor(30, 30, 30))
            return
        h_map, w_map = grid.shape
        resolution = self.mapper.map.resolution
        offset = self.mapper.map.offset
        self._grid_image_buffer = np.full((h_map, w_map), 0xFF808080, dtype=np.uint32)
        self._grid_image_buffer[grid == 0] = 0xFF000000
        self._grid_image_buffer[grid == 255] = 0xFFFFFFFF
        image = QImage(
            self._grid_image_buffer.tobytes(),
            w_map, h_map, w_map * 4, QImage.Format_ARGB32
        )
        wx_tl, wy_tl = self.mapper.map.map_to_world(0, 0)
        sx_tl, sy_tl = self.world_to_screen(wx_tl, wy_tl, w, h)
        wx_br, wy_br = self.mapper.map.map_to_world(w_map - 1, h_map - 1)
        sx_br, sy_br = self.world_to_screen(wx_br, wy_br, w, h)
        map_rect_x = min(sx_tl, sx_br)
        map_rect_y = min(sy_tl, sy_br)
        map_rect_w = abs(sx_br - sx_tl)
        map_rect_h = abs(sy_br - sy_tl)
        if map_rect_w > 10 and map_rect_h > 10:
            scaled = image.scaled(map_rect_w, map_rect_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            painter.drawImage(map_rect_x, map_rect_y, scaled)
        else:
            scaled = image.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            px = (w - scaled.width()) // 2
            py = (h - scaled.height()) // 2
            painter.drawImage(px, py, scaled)

    # ============================================================
    # 坐标轴和网格
    # ============================================================
    def _draw_axes_and_grid(self, painter, w, h):
        cx = w // 2 + int(self.axis_offset_x * w)
        cy = int(h * self.axis_offset_y)
        pen = QPen(QColor(100, 100, 100, 80))
        pen.setWidth(1)
        painter.setPen(pen)
        grid_step = int(500 * self.scale)
        for i in range(-80, 81):
            x = cx + i * grid_step
            painter.drawLine(x, 0, x, h)
            y = cy + i * grid_step
            painter.drawLine(0, y, w, y)
        pen = QPen(QColor(150, 150, 150, 150))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawLine(cx, 0, cx, h)
        painter.drawLine(0, cy, w, cy)
        painter.setPen(QColor(200, 200, 200))
        font = QFont("Microsoft YaHei", 8)
        painter.setFont(font)
        for i in range(0, 2001, 500):
            sy = cy - int(i * self.scale)
            if 0 <= sy <= h:
                painter.drawText(cx + 5, sy, f"{i/10:.0f}cm")
                painter.drawLine(cx - 3, sy, cx + 3, sy)
        for i in range(-1000, 1001, 500):
            sx = cx - int(i * self.scale)
            if 0 <= sx <= w:
                painter.drawText(sx, cy + 15, f"{i/10:.0f}cm")
                painter.drawLine(sx, cy - 3, sx, cy + 3)

    # ============================================================
    # 历史点云
    # ============================================================
    def _draw_history_points(self, painter, w, h):
        if self.navigator and self.navigator.is_active():
            return
        history = self.mapper.get_all_scanned_points()
        if not history:
            return
        pen = QPen(QColor(255, 80, 80, 100))
        painter.setPen(pen)
        for wx, wy in history:
            sx, sy = self.world_to_screen(wx, wy, w, h)
            if 0 <= sx < w and 0 <= sy < h:
                painter.drawPoint(sx, sy)

    # ============================================================
    # 最新帧点云
    # ============================================================
    def _draw_latest_points(self, painter, w, h):
        local_pts, world_pts = self.mapper.get_latest_points()
        if not world_pts:
            return
        pen = QPen(QColor(255, 50, 50))
        pen.setWidth(2)
        painter.setPen(pen)
        for wx, wy in world_pts:
            sx, sy = self.world_to_screen(wx, wy, w, h)
            if 0 <= sx < w and 0 <= sy < h:
                painter.drawPoint(sx, sy)

    # ============================================================
    # 机器人轨迹
    # ============================================================
    def _draw_trajectory(self, painter, w, h):
        trajectory = self.mapper.get_trajectory()
        if len(trajectory) <= 1:
            return
        pen = QPen(QColor(50, 255, 50))
        pen.setWidth(2)
        painter.setPen(pen)
        for i in range(len(trajectory) - 1):
            x1, y1 = self.world_to_screen(trajectory[i][0], trajectory[i][1], w, h)
            x2, y2 = self.world_to_screen(trajectory[i+1][0], trajectory[i+1][1], w, h)
            if (0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h):
                painter.drawLine(x1, y1, x2, y2)

    # ============================================================
    # 导航路径和目标点
    # ============================================================
    def _draw_navigation(self, painter, w, h):
        if not self.navigator or not self.navigator.is_active():
            return
        waypoints = self.navigator.get_waypoints()
        if len(waypoints) > 1:
            pen = QPen(QColor(0, 255, 255))
            pen.setWidth(3)
            painter.setPen(pen)
            for i in range(len(waypoints) - 1):
                x1, y1 = self.world_to_screen(waypoints[i][0], waypoints[i][1], w, h)
                x2, y2 = self.world_to_screen(waypoints[i+1][0], waypoints[i+1][1], w, h)
                painter.drawLine(x1, y1, x2, y2)
        if waypoints:
            tx, ty = waypoints[-1]
            tsx, tsy = self.world_to_screen(tx, ty, w, h)
            pen = QPen(QColor(0, 255, 0))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.setBrush(QColor(0, 255, 0, 100))
            painter.drawEllipse(tsx - 10, tsy - 10, 20, 20)
            painter.setPen(QColor(0, 255, 0))
            font = QFont("Microsoft YaHei", 10, QFont.Bold)
            painter.setFont(font)
            painter.drawText(tsx + 15, tsy, f"目标 ({tx:.0f}, {ty:.0f})")

    # ============================================================
    # 机器人当前位置（显示真实车身尺寸 + 安全边界）
    # ============================================================
    def _draw_robot(self, painter, w, h):
        stats = self.mapper.get_stats()
        rx, ry = stats['pose'][0], stats['pose'][1]
        sx, sy = self.world_to_screen(rx, ry, w, h)
        theta = stats['pose'][2]
        # 1. 车身实际尺寸：270mm 正方形（半透明青色）
        half_w = int((self.navigator.robot_width_mm / 2.0) * self.scale) if self.navigator else int(135 * self.scale)
        painter.save()
        painter.translate(sx, sy)
        painter.rotate(-math.degrees(theta))
        pen = QPen(QColor(0, 200, 255, 200))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(QColor(0, 200, 255, 60))
        painter.drawRect(-half_w, -half_w, half_w * 2, half_w * 2)
        painter.restore()
        # 2. 安全圆半径 = robot_radius_mm（135mm，车身半宽）（虚线青色）
        r_safe = int((self.navigator.robot_radius_mm if self.navigator else 135) * self.scale)
        if r_safe > 5:
            pen = QPen(QColor(0, 255, 200, 150))
            pen.setWidth(2)
            pen.setStyle(Qt.DotLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(sx - r_safe, sy - r_safe, 2 * r_safe, 2 * r_safe)
        # 3. 中心点 + 朝向箭头
        pen = QPen(QColor(50, 150, 255))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(QColor(50, 150, 255, 220))
        painter.drawEllipse(sx - 6, sy - 6, 12, 12)
        arrow_len = 25
        dx = int(-arrow_len * math.sin(theta))
        dy = int(-arrow_len * math.cos(theta))
        pen.setWidth(3)
        painter.setPen(pen)
        painter.drawLine(sx, sy, sx + dx, sy + dy)

    # ============================================================
    # 文字信息
    # ============================================================
    def _draw_info_text(self, painter, w, h):
        stats = self.mapper.get_stats()
        local_pts, world_pts = self.mapper.get_latest_points()
        theta = stats['pose'][2]
        painter.setPen(QColor(220, 220, 220))
        font = QFont("Microsoft YaHei", 10)
        painter.setFont(font)
        info = (f"帧数: {stats['frame_count']} "
                f"当前帧点数: {len(local_pts)} "
                f"历史点数: {stats['history_points']}"
                f"机器人: ({stats['pose'][0]:.0f}, {stats['pose'][1]:.0f})mm "
                f"朝向: {math.degrees(theta):.1f}°")
        painter.drawText(10, 25, info)
        if self.navigator:
            nav_status = self.navigator.get_status()
            painter.setPen(QColor(0, 255, 170))
            font = QFont("Microsoft YaHei", 11, QFont.Bold)
            painter.setFont(font)
            painter.drawText(10, 50, nav_status)

    # ============================================================
    # 膨胀障碍物栅格
    # ============================================================
    def _draw_inflated_grid(self, painter, w, h):
        if not self.navigator:
            return
        grid_data, size, resolution, offset = self.navigator.get_inflated_grid()
        if grid_data is None or grid_data.size == 0:
            return
        self._inflated_image_buffer = np.zeros((size, size), dtype=np.uint32)
        self._inflated_image_buffer[grid_data] = 0x78FF1414
        image = QImage(
            self._inflated_image_buffer.tobytes(),
            size, size, size * 4, QImage.Format_ARGB32
        )
        wx_tl, wy_tl = self.mapper.map.map_to_world(0, 0)
        sx_tl, sy_tl = self.world_to_screen(wx_tl, wy_tl, w, h)
        wx_br, wy_br = self.mapper.map.map_to_world(size - 1, size - 1)
        sx_br, sy_br = self.world_to_screen(wx_br, wy_br, w, h)
        map_rect_x = min(sx_tl, sx_br)
        map_rect_y = min(sy_tl, sy_br)
        map_rect_w = abs(sx_br - sx_tl)
        map_rect_h = abs(sy_br - sy_tl)
        if map_rect_w > 10 and map_rect_h > 10:
            scaled = image.scaled(map_rect_w, map_rect_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            painter.drawImage(map_rect_x, map_rect_y, scaled)

    # ============================================================
    # 原始 A* 路径（灰色虚线）
    # ============================================================
    def _draw_raw_path(self, painter, w, h):
        if not self.navigator:
            return
        raw_path = self.navigator.get_raw_path()
        if len(raw_path) < 2:
            return
        pen = QPen(QColor(160, 160, 160, 180))
        pen.setWidth(1)
        pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        for i in range(len(raw_path) - 1):
            x1, y1 = self.world_to_screen(raw_path[i][0], raw_path[i][1], w, h)
            x2, y2 = self.world_to_screen(raw_path[i+1][0], raw_path[i+1][1], w, h)
            painter.drawLine(x1, y1, x2, y2)

    # ============================================================
    # 路径点标记
    # ============================================================
    def _draw_waypoint_markers(self, painter, w, h):
        if not self.navigator:
            return
        waypoints = self.navigator.get_waypoints()
        if not waypoints:
            return
        pen = QPen(QColor(0, 200, 255))
        pen.setWidth(1)
        painter.setPen(pen)
        painter.setBrush(QColor(0, 200, 255, 80))
        font = QFont("Microsoft YaHei", 7)
        painter.setFont(font)
        for i, (wx, wy) in enumerate(waypoints):
            sx, sy = self.world_to_screen(wx, wy, w, h)
            if 0 <= sx < w and 0 <= sy < h:
                if i == self.navigator.current_wp:
                    painter.setBrush(QColor(255, 255, 0, 180))
                    painter.drawEllipse(sx - 6, sy - 6, 12, 12)
                    painter.setBrush(QColor(0, 200, 255, 80))
                else:
                    painter.drawEllipse(sx - 4, sy - 4, 8, 8)
                if i % 3 == 0 or i == len(waypoints) - 1:
                    painter.setPen(QColor(200, 240, 255))
                    painter.drawText(sx + 6, sy - 6, str(i))
                    painter.setPen(pen)

    # ============================================================
    # Lookahead 点
    # ============================================================
    def _draw_lookahead(self, painter, w, h):
        if not self.navigator:
            return
        lp = self.navigator.get_lookahead_point()
        if lp is None:
            return
        sx, sy = self.world_to_screen(lp[0], lp[1], w, h)
        if not (0 <= sx < w and 0 <= sy < h):
            return
        pen = QPen(QColor(0, 255, 100))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(sx - 12, sy - 12, 24, 24)
        painter.drawLine(sx - 16, sy, sx + 16, sy)
        painter.drawLine(sx, sy - 16, sx, sy + 16)
        font = QFont("Microsoft YaHei", 8, QFont.Bold)
        painter.setFont(font)
        painter.setPen(QColor(0, 255, 100))
        painter.drawText(sx + 18, sy, "LP")

    # ============================================================
    # 安全距离圈 + 障碍物扇区（仅导航激活时显示）
    # ============================================================
    def _draw_safety_zone(self, painter, w, h):
        if not self.navigator or not self.navigator.is_active():
            return
        stats = self.mapper.get_stats()
        rx, ry = stats['pose'][0], stats['pose'][1]
        theta = stats['pose'][2]
        sx, sy = self.world_to_screen(rx, ry, w, h)
        # 400mm 警戒圈
        pen = QPen(QColor(255, 60, 60, 180))
        pen.setWidth(1)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        r400 = int(400 * self.scale)
        if r400 > 5:
            painter.drawArc(sx - r400, sy - r400, r400 * 2, r400 * 2, int((90 - math.degrees(theta) - 45) * 16), int(90 * 16))
        # 200mm 紧急圈
        pen = QPen(QColor(255, 30, 30, 220))
        pen.setWidth(2)
        painter.setPen(pen)
        r200 = int(200 * self.scale)
        if r200 > 3:
            painter.drawArc(sx - r200, sy - r200, r200 * 2, r200 * 2, int((90 - math.degrees(theta) - 25) * 16), int(50 * 16))
        # 障碍物扇区
        sectors = self.navigator.get_obstacle_sectors()
        if sectors:
            def draw_sector(rad_start, rad_end, color, max_dist_mm):
                r = int(max_dist_mm * self.scale)
                if r < 3:
                    return
                start_deg = 90 - math.degrees(theta + rad_end)
                span_deg = math.degrees(rad_end - rad_start)
                painter.setBrush(color)
                painter.setPen(Qt.NoPen)
                painter.drawPie(sx - r, sy - r, r * 2, r * 2, int(start_deg * 16), int(span_deg * 16))
            if sectors.get('front', float('inf')) < 400:
                draw_sector(math.radians(-25), math.radians(25), QColor(255, 0, 0, 60), 400)
            if sectors.get('f_left', float('inf')) < 400 or sectors.get('f_right', float('inf')) < 400:
                draw_sector(math.radians(25), math.radians(70), QColor(255, 160, 0, 45), 400)
                draw_sector(math.radians(-70), math.radians(-25), QColor(255, 160, 0, 45), 400)
        # 最近障碍物连线
        _, world_pts = self.mapper.get_latest_points()
        if world_pts:
            pen = QPen(QColor(255, 80, 80, 150))
            pen.setWidth(1)
            painter.setPen(pen)
            for wx, wy in world_pts:
                dx = wx - rx
                dy = wy - ry
                dist = math.hypot(dx, dy)
                if dist > 1000 or dist < 80:
                    continue
                angle = math.atan2(dy, dx) - theta
                while angle > math.pi:
                    angle -= 2 * math.pi
                while angle < -math.pi:
                    angle += 2 * math.pi
                if abs(angle) < math.radians(70):
                    ex, ey = self.world_to_screen(wx, wy, w, h)
                    painter.drawLine(sx, sy, ex, ey)

    # ============================================================
    # 导航调试文字叠加
    # ============================================================
    def _draw_nav_debug_overlay(self, painter, w, h):
        if not self.navigator:
            return
        lines = []
        lines.append(f"状态: {self.navigator.state}")
        lines.append(f"障碍级别: {self.navigator.get_obstacle_level()}")
        sectors = self.navigator.get_obstacle_sectors()
        if sectors:
            def fmt(v):
                return f"{v:.0f}" if v < 9999 else "∞"
            lines.append(f"前方:{fmt(sectors.get('front',9999))} " 
                         f"左前:{fmt(sectors.get('f_left',9999))} " 
                         f"右前:{fmt(sectors.get('f_right',9999))}")
            lines.append(f"左侧:{fmt(sectors.get('left',9999))} " 
                         f"右侧:{fmt(sectors.get('right',9999))}")
        lp = self.navigator.get_lookahead_point()
        if lp:
            lines.append(f"Lookahead: ({lp[0]:.0f}, {lp[1]:.0f})")
        grid_data, *_ = self.navigator.get_inflated_grid()
        if grid_data is not None:
            occ_count = int(np.count_nonzero(grid_data))
            lines.append(f"膨胀栅格: {occ_count}")
        permanent_cells = len(self.navigator.get_stable_obstacle_cells())
        lines.append(f"永久障碍物栅格: {permanent_cells}")
        margin = self.navigator.obstacle_margin
        margin_mm = margin * self.mapper.map.resolution
        lines.append(f"膨胀半径: {margin_mm:.0f} mm ({margin} 栅格)")
        if self.navigator:
            lines.append(f"车身: {self.navigator.robot_width_mm:.0f}mm 安全圆: {self.navigator.robot_radius_mm:.0f}mm")
            lines.append(f"最小通道: {self.navigator.robot_width_mm + 2*margin_mm:.0f}mm")
        painter.setPen(QColor(255, 220, 100))
        font = QFont("Microsoft YaHei", 9, QFont.Bold)
        painter.setFont(font)
        y_off = 70
        for line in lines:
            painter.drawText(10, y_off, line)
            y_off += 16

    # ============================================================
    # 备用视图
    # ============================================================
    def _draw_pointcloud(self, painter, w, h):
        painter.fillRect(self.rect(), QColor(30, 30, 30))
        self._draw_axes_and_grid(painter, w, h)
        self._draw_history_points(painter, w, h)
        self._draw_latest_points(painter, w, h)
        self._draw_trajectory(painter, w, h)
        self._draw_navigation(painter, w, h)
        self._draw_robot(painter, w, h)
        self._draw_info_text(painter, w, h)

    def _draw_gridmap(self, painter, w, h):
        grid = self.mapper.get_map_display()
        if grid is None or grid.size == 0:
            return
        h_map, w_map = grid.shape
        self._grid_image_buffer = np.full((h_map, w_map), 0xFF808080, dtype=np.uint32)
        self._grid_image_buffer[grid == 0] = 0xFF000000
        self._grid_image_buffer[grid == 255] = 0xFFFFFFFF
        image = QImage(
            self._grid_image_buffer.tobytes(),
            w_map, h_map, w_map * 4, QImage.Format_ARGB32
        )
        scaled = image.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        px = (w - scaled.width()) // 2
        py = (h - scaled.height()) // 2
        painter.drawImage(px, py, scaled)
        scale_x = scaled.width() / w_map
        scale_y = scaled.height() / h_map
        stats = self.mapper.get_stats()
        rx, ry = stats['pose'][0], stats['pose'][1]
        mx, my = self.mapper.map.world_to_map(rx, ry)
        sx = px + int(mx * scale_x)
        sy = py + int(my * scale_y)
        pen = QPen(QColor(255, 0, 0))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(QColor(255, 0, 0, 200))
        painter.drawEllipse(sx - 5, sy - 5, 10, 10)
        theta = stats['pose'][2]
        dx = int(-15 * math.sin(theta))
        dy = int(-15 * math.cos(theta))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawLine(sx, sy, sx + dx, sy + dy)
