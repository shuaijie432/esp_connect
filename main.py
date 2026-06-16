


"""Main: 静态地图导航 + 速度闭环控制下发 - 最终修复版（适配机器人半径230mm）"""

import sys
import math
import time
import threading
import warnings
import uuid
from queue import Queue, Empty

import paho.mqtt.client as mqtt

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSplitter, QLineEdit
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QFont

from lidar_parser import parse_frame
from mapper import LidarMapper
from map_widget import MapWidget
from navigator import Navigator
from laser_odometry import LaserOdometry

MQTT_HOST = "10.113.145.227"
MQTT_PORT = 1883
MQTT_USER = "esp_send"
MQTT_PASS = "00000000"
TOPIC_LIDAR = "esp/f79541/data"
TOPIC_CONTROL = "device/f79541/data"
TOPIC_OPENMV  = "openmv/nav"

MAP_SIZE_MM = 8000
RESOLUTION_MM = 15
FRAME_QUEUE_SIZE = 3
MAP_FILE = "1.png"


class Communicate(QObject):
    new_data = pyqtSignal()
    new_odom = pyqtSignal()
    status_msg = pyqtSignal(str, str)
    obstacle_fusion = pyqtSignal(list)


class MainWindow(QMainWindow):
    def __init__(self, mapper: LidarMapper, comm: Communicate, client: mqtt.Client = None):
        super().__init__()
        self.mapper = mapper
        self.comm = comm
        self.client = client
        self.setWindowTitle("激光雷达静态地图导航")
        self.setGeometry(50, 50, 1920, 1080)
        self.showMaximized()

        self.navigator = Navigator(mapper)
        self.laser_odom = LaserOdometry(num_particles=300)
        if mapper.static_map_mode and mapper.map is not None:
            self.laser_odom.set_map(mapper.map)

        self._last_wheel_x = 0.0
        self._last_wheel_y = 0.0
        self._last_wheel_theta = 0.0
        self._wheel_initialized = False

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(10)

        self.map_view = MapWidget(mapper, navigator=self.navigator, mode="combined")
        self.map_view.setMinimumHeight(800)
        left_layout.addWidget(QLabel("静态地图 + 实时点云 + 导航路径 + 永久障碍物"))
        left_layout.addWidget(self.map_view, 1)

        splitter.addWidget(left_widget)

        right_widget = QWidget()
        right_widget.setMaximumWidth(320)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(15)

        self.status_label = QLabel("状态: 等待连接...")
        self.status_label.setStyleSheet("color: orange; font-size: 16px; font-weight: bold;")
        right_layout.addWidget(self.status_label)

        self.loop_label = QLabel("闭环: 未检测")
        self.loop_label.setStyleSheet("font-size: 14px; color: #ff6666; font-weight: bold;")
        right_layout.addWidget(self.loop_label)

        self.stats_label = QLabel("帧数: 0\n点数: 0")
        self.stats_label.setStyleSheet("font-size: 13px; line-height: 1.5;")
        right_layout.addWidget(self.stats_label)

        self.odom_label = QLabel("里程计: 等待数据...")
        self.odom_label.setStyleSheet("font-size: 13px; line-height: 1.5; color: #66ccff;")
        right_layout.addWidget(self.odom_label)

        self.corr_label = QLabel("修正量: dx=0, dy=0, dθ=0°")
        self.corr_label.setStyleSheet("font-size: 12px; line-height: 1.5; color: #ffcc66;")
        right_layout.addWidget(self.corr_label)

        self.vel_label = QLabel("速度: v=0 mm/s, w=0 rad/s")
        self.vel_label.setStyleSheet("font-size: 13px; line-height: 1.5; color: #ffcc66;")
        right_layout.addWidget(self.vel_label)

        self.queue_label = QLabel("队列: 0/3")
        self.queue_label.setStyleSheet("font-size: 13px; color: #ff6666;")
        right_layout.addWidget(self.queue_label)

        self.nav_debug_label = QLabel("导航调试: 等待...")
        self.nav_debug_label.setStyleSheet("font-size: 11px; line-height: 1.4; color: #aaaaff;")
        right_layout.addWidget(self.nav_debug_label)

        self.obstacle_label = QLabel("障碍物: 0个永久, 0个临时")
        self.obstacle_label.setStyleSheet("font-size: 11px; line-height: 1.4; color: #ffaa66;")
        right_layout.addWidget(self.obstacle_label)

        right_layout.addStretch()

        self.btn_save = QPushButton("保存地图")
        self.btn_save.setStyleSheet("font-size: 14px; padding: 10px;")
        self.btn_save.clicked.connect(self.save_map)
        right_layout.addWidget(self.btn_save)

        self.btn_clear = QPushButton("清空地图")
        self.btn_clear.setStyleSheet("font-size: 14px; padding: 10px;")
        self.btn_clear.clicked.connect(self.clear_map)
        right_layout.addWidget(self.btn_clear)

        self.btn_reset_odom = QPushButton("重置里程计")
        self.btn_reset_odom.setStyleSheet("font-size: 14px; padding: 10px;")
        self.btn_reset_odom.clicked.connect(self.reset_odom)
        right_layout.addWidget(self.btn_reset_odom)

        right_layout.addSpacing(20)
        nav_title = QLabel("▎目标点导航")
        nav_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #00ffaa;")
        right_layout.addWidget(nav_title)

        self.target_x = QLineEdit("0")
        self.target_x.setStyleSheet(
            "font-size: 12px; padding: 5px; background-color: #2a2a2a; color: #fff; border: 1px solid #555;"
        )
        right_layout.addWidget(QLabel("目标 X (mm):"))
        right_layout.addWidget(self.target_x)

        self.target_y = QLineEdit("1000")
        self.target_y.setStyleSheet(
            "font-size: 12px; padding: 5px; background-color: #2a2a2a; color: #fff; border: 1px solid #555;"
        )
        right_layout.addWidget(QLabel("目标 Y (mm):"))
        right_layout.addWidget(self.target_y)

        self.target_theta_deg = QLineEdit("0")
        self.target_theta_deg.setStyleSheet(
            "font-size: 12px; padding: 5px; background-color: #2a2a2a; color: #fff; border: 1px solid #555;"
        )
        right_layout.addWidget(QLabel("目标角度 (°):"))
        right_layout.addWidget(self.target_theta_deg)

        self.btn_nav = QPushButton("🚀 开始导航")
        self.btn_nav.setStyleSheet(
            "font-size: 14px; padding: 10px; background-color: #00aa66; color: white; font-weight: bold;"
        )
        self.btn_nav.clicked.connect(self.start_navigation)
        right_layout.addWidget(self.btn_nav)

        self.btn_stop_nav = QPushButton("⏹ 停止导航")
        self.btn_stop_nav.setStyleSheet(
            "font-size: 14px; padding: 10px; background-color: #aa3333; color: white; font-weight: bold;"
        )
        self.btn_stop_nav.clicked.connect(self.stop_navigation)
        right_layout.addWidget(self.btn_stop_nav)

        self.btn_send_ack = QPushButton("📤 手动发送完成帧")
        self.btn_send_ack.setStyleSheet(
            "font-size: 13px; padding: 8px; background-color: #3366aa; color: white; font-weight: bold;"
        )
        self.btn_send_ack.clicked.connect(self.on_manual_send_ack)
        right_layout.addWidget(self.btn_send_ack)

        # right_layout.addSpacing(10)
        # hold_title = QLabel("▎速度持续控制 (按住发送，松开停止)")
        # hold_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #ffcc00;")
        # right_layout.addWidget(hold_title)
        #
        # self.vx_input = QLineEdit("100")
        # self.vy_input = QLineEdit("100")
        # self.vw_input = QLineEdit("30")
        # for inp in (self.vx_input, self.vy_input, self.vw_input):
        #     inp.setStyleSheet(
        #         "font-size: 12px; padding: 5px; background-color: #2a2a2a; color: #fff; border: 1px solid #555;"
        #     )
        #
        # self.btn_hold_all = QPushButton("🚀 长按发送速度")
        # self.btn_hold_all.setStyleSheet(
        #     "font-size: 14px; padding: 10px; background-color: #ff8800; color: white; font-weight: bold;"
        # )
        # self.btn_hold_all.pressed.connect(self.on_hold_pressed)
        # self.btn_hold_all.released.connect(self.on_hold_released)
        #
        # right_layout.addWidget(QLabel("vx 速度 (mm/s):"))
        # right_layout.addWidget(self.vx_input)
        # right_layout.addWidget(QLabel("vy 速度 (mm/s):"))
        # right_layout.addWidget(self.vy_input)
        # right_layout.addWidget(QLabel("vw 速度 (°/s):"))
        # right_layout.addWidget(self.vw_input)
        # right_layout.addWidget(self.btn_hold_all)

        self.nav_status = QLabel("导航: 空闲")
        self.nav_status.setStyleSheet("font-size: 13px; color: #00ffaa;")
        right_layout.addWidget(self.nav_status)

        right_layout.addStretch()
        splitter.addWidget(right_widget)
        splitter.setSizes([1600, 320])

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_display)
        self.timer.start(200)

        self.comm.new_data.connect(self.update_display)
        self.comm.new_odom.connect(self.update_odom_display)
        self.comm.status_msg.connect(self.set_status)
        self.comm.obstacle_fusion.connect(self._on_obstacle_fusion)

        self.nav_timer = QTimer()
        self.nav_timer.timeout.connect(self.nav_step)
        self.nav_timer.start(100)

        self._last_vx = 0.0
        self._last_vy = 0.0
        self._last_vw = 0.0
        self._last_cmd_time = 0.0
        self._nav_was_active = False

        self._obstacle_fusion_interval = 0.5
        self._last_obstacle_fusion = 0.0

        self._hold_axes = {"vx": 0.0, "vy": 0.0, "vw": 0.0}
        self._hold_timer = QTimer()
        self._hold_timer.timeout.connect(self.on_hold_tick)
        self._hold_timer.start(100)

        self._need_replan = False
        self._alignment_ack_done = False
        self._pending_obstacles = []
        self._obstacle_processing = False

    def _on_obstacle_fusion(self, obstacle_candidates):
        self._pending_obstacles.extend(obstacle_candidates)
        if len(self._pending_obstacles) > 5000:
            self._pending_obstacles = self._pending_obstacles[-2500:]

    def _process_pending_obstacles(self):
        if not self._pending_obstacles:
            return
        candidates = self._pending_obstacles.copy()
        self._pending_obstacles.clear()

        try:
            result = self.navigator.permanently_add_obstacles(candidates)
            permanent_count = sum(1 for obs in self.navigator._stable_obstacles.values()
                                 if obs.get('is_permanent', False))
            temp_count = len(self.navigator._stable_obstacles) - permanent_count
            self.obstacle_label.setText(
                f"障碍物: {permanent_count}个永久, {temp_count}个临时"
            )
            if result.get('updated', 0) > 0 and self.navigator.is_active():
                print(f"[MAP_FUSION] 障碍物位置变化，触发重规划")
                self._need_replan = True
        except Exception as e:
            print(f"[ERROR] 障碍物融合失败: {e}")
            import traceback
            traceback.print_exc()

    def start_navigation(self):
        # ===== 可选：检查地图是否有足够障碍物信息 =====
        # 注释掉，用户可按需取消注释
        # occ_count = np.count_nonzero(self.mapper.map.log_odds > self.mapper.map.occ_thresh)
        # if occ_count < 100:
        #     self.set_status("地图障碍物信息不足，请等待激光雷达扫描一圈后再试", "orange")
        #     return
        # ==============================================

        try:
            tx = float(self.target_x.text())
            ty = float(self.target_y.text())
            ttheta_deg = float(self.target_theta_deg.text())
            ttheta_rad = math.radians(ttheta_deg)

            success = self.navigator.set_target(tx, ty, ttheta_rad)

            if success:
                self.set_status(f"开始导航 → ({tx:.0f}, {ty:.0f}) @{ttheta_deg:.0f}°", "green")
                self.nav_status.setText(self.navigator.get_status())
                waypoints = self.navigator.get_waypoints()
                self.nav_debug_label.setText(
                    f"路径点: {len(waypoints)}个\n"
                    f"起点: ({self.mapper.pose.x:.0f}, {self.mapper.pose.y:.0f})\n"
                    f"终点: ({tx:.0f}, {ty:.0f})\n"
                    f"目标朝向: {ttheta_deg:.0f}°"
                )
            else:
                self.set_status("路径规划失败（目标不可达）", "red")
                self.nav_debug_label.setText(
                    f"A* 失败: 起点=({self.mapper.pose.x:.0f},{self.mapper.pose.y:.0f})\n"
                    f"终点=({tx:.0f},{ty:.0f})"
                )
        except ValueError:
            self.set_status("请输入有效的数字坐标/角度", "red")

    def send_alignment_ack_frame(self):
        if self.client is None or not self.client.is_connected():
            return

        frame = bytearray()
        frame.append(0xAA)
        frame.append(0x55)
        frame.append(0x01)
        frame.append(0xA1)
        checksum = sum(frame[2:]) & 0xFF
        frame.append(checksum)
        frame.append(0xBB)

        try:
            self.client.publish(TOPIC_CONTROL, bytes(frame), qos=1)
            hex_str = ' '.join(f'{b:02X}' for b in frame)
            print(f"[CMD] 导航完成校验帧已发送(ESP32) | HEX[{hex_str}]")
        except Exception as e:
            print(f"[CMD] 发送失败: {e}")

        try:
            self.client.publish(TOPIC_OPENMV, "1", qos=1)
            print(f"[CMD] 角度对准标志位已发送(OpenMV) -> {TOPIC_OPENMV}")
        except Exception as e:
            print(f"[CMD] OpenMV发送失败: {e}")

        self._alignment_ack_done = True

    def on_manual_send_ack(self):
        self.send_alignment_ack_frame()
        self._alignment_ack_done = True
        self.navigator.cancel()
        self.navigator._align_settle_until = 0.0
        self.navigator._alignment_ack_sent = False
        self.navigator._send_alignment_ack = False
        self.navigator._align_stable_count = 0
        self.nav_status.setText("导航: 手动发送完成帧")
        self.nav_status.setStyleSheet("font-size: 13px; color: #3366aa;")
        self.set_status("已手动发送导航完成校验帧，速度下发已停止", "blue")

    def on_hold_pressed(self):
        try:
            vx = float(self.vx_input.text())
            vy = float(self.vy_input.text())
            vw = float(self.vw_input.text())
        except ValueError:
            self.set_status("速度输入无效，请输入数字", "red")
            return
        self._hold_axes = {"vx": vx, "vy": vy, "vw": vw}
        self.set_status(f"持续发送 vx={vx:.1f} vy={vy:.1f} vw={vw:.1f}", "green")

    def on_hold_released(self):
        self._hold_axes = {"vx": 0.0, "vy": 0.0, "vw": 0.0}
        self.send_velocity_command(0.0, 0.0, 0.0)
        self.set_status("速度发送已停止", "orange")

    def on_hold_tick(self):
        vx = self._hold_axes.get("vx", 0.0)
        vy = self._hold_axes.get("vy", 0.0)
        vw = self._hold_axes.get("vw", 0.0)
        if vx != 0.0 or vy != 0.0 or vw != 0.0:
            self.send_velocity_command(vx, vy, vw)

    def stop_navigation(self):
        self.navigator.cancel()
        self.send_velocity_command(0.0, 0.0, 0.0)
        self.nav_status.setText("导航: 已停止")
        self.nav_status.setStyleSheet("font-size: 13px; color: #ff6666;")
        self.set_status("导航已停止", "orange")

    def nav_step(self):
        self._process_pending_obstacles()

        is_active = self.navigator.is_active()
        send_ack = self.navigator._send_alignment_ack
        if send_ack:
            self.navigator._send_alignment_ack = False

        if not is_active:
            if self._nav_was_active:
                self.send_velocity_command(0.0, 0.0, 0.0)
                self._nav_was_active = False
            return

        self._nav_was_active = True

        with self.mapper.lock:
            x, y, theta = self.mapper.pose.x, self.mapper.pose.y, self.mapper.pose.theta

        if self._need_replan:
            self._need_replan = False
            if self.navigator.waypoints:
                target = self.navigator.waypoints[-1]
                target_theta = self.navigator.target_theta
                old_state = self.navigator.state
                old_wp = self.navigator.current_wp

                success = self.navigator.set_target(target[0], target[1], target_theta)
                if success:
                    self.navigator.current_wp = min(old_wp, len(self.navigator.waypoints) - 1)
                    self.set_status("检测到新障碍物，已重新规划路径", "orange")
                    print(f"[REPLAN] 恢复路径点 {self.navigator.current_wp}/{len(self.navigator.waypoints)}")
                else:
                    self.set_status("重规划失败！目标不可达", "red")
                    self.navigator.state = old_state

        vx, vy, vw = self.navigator.update(x, y, theta)
        status_text = self.navigator.get_status()
        wp = self.navigator.get_waypoints()
        curr = self.navigator.current_wp

        if send_ack:
            self.send_alignment_ack_frame()
            self.navigator.state = "DONE"
            print(f"[NAV] 角度对准完成！最终角度: {math.degrees(theta):.1f}°")

        now = time.time()
        if (now - self._last_cmd_time) > 0.1 or \
           abs(vx - self._last_vx) > 15 or \
           abs(vy - self._last_vy) > 15 or \
           abs(vw - self._last_vw) > 0.08:
            self.send_velocity_command(vx, vy, vw)
            self._last_vx = vx
            self._last_vy = vy
            self._last_vw = vw
            self._last_cmd_time = now

        self.nav_status.setText(status_text)
        if wp:
            self.nav_debug_label.setText(
                f"路径点: {curr}/{len(wp)}\n"
                f"当前: ({x:.0f}, {y:.0f})\n"
                f"目标: ({wp[-1][0]:.0f}, {wp[-1][1]:.0f})\n"
                f"速度: vx={vx:.0f} vy={vy:.0f} vw={math.degrees(vw):.1f}°/s"
            )

    def send_velocity_command(self, vx: float, vy: float, vw: float):
        if self.client is None or not self.client.is_connected():
            return

        try:
            import struct
            vx_m = vx / 1000.0
            vy_m = vy / 1000.0
            data = struct.pack('<fff', vx_m, -vy_m, vw)
            length = len(data)

            frame = bytearray()
            frame.append(0xAA)
            frame.append(0x55)
            frame.append(length)
            frame.extend(data)
            checksum = sum(frame[2:]) & 0xFF
            frame.append(checksum)
            frame.append(0xBB)

            self.client.publish(TOPIC_CONTROL, bytes(frame), qos=0)

            if not hasattr(self, '_cmd_cnt'):
                self._cmd_cnt = 0
            self._cmd_cnt += 1
            if self._cmd_cnt % 20 == 0:
                hex_str = ' '.join(f'{b:02X}' for b in frame)
                print(f"[CMD] 速度: vx={vx:6.1f} vy={vy:6.1f} vw={math.degrees(vw):5.1f}°/s | HEX[{hex_str}]")

        except Exception as e:
            print(f"[CMD] 速度帧发送失败: {e}")

    def update_display(self):
        self.map_view.update()

        stats = self.mapper.get_stats()
        self.stats_label.setText(
            f"帧数: {stats['frame_count']}\n"
            f"总点数: {stats['total_points']}\n"
            f"历史点数: {stats['history_points']}\n"
            f"地图: {stats['map_size']}x{stats['map_size']} ({RESOLUTION_MM}mm/格)\n"
            f"机器人: ({stats['pose'][0]:.0f}, {stats['pose'][1]:.0f})mm\n"
            f"朝向: {math.degrees(stats['pose'][2]):.1f}°"
        )

        if stats['loop_detected']:
            self.loop_label.setText("闭环: ✓ 已闭合")
            self.loop_label.setStyleSheet("font-size: 14px; color: #66ff66; font-weight: bold;")
        else:
            self.loop_label.setText("闭环: ✗ 未闭合")
            self.loop_label.setStyleSheet("font-size: 14px; color: #ff6666; font-weight: bold;")

    def update_odom_display(self):
        odom_stats = self.mapper.get_odom_stats()
        self.odom_label.setText(
            f"里程计积分\n"
            f"X: {odom_stats['x']:.1f}mm\n"
            f"Y: {odom_stats['y']:.1f}mm\n"
            f"θ: {math.degrees(odom_stats['theta']):.2f}°\n"
            f"轨迹点数: {odom_stats['trajectory_len']}"
        )
        corr = odom_stats['loop_correction']
        self.corr_label.setText(
            f"修正量: dx={corr[0]:.1f}, dy={corr[1]:.1f}, "
            f"dθ={math.degrees(corr[2]):.1f}°"
        )
        self.vel_label.setText(
            f"最新速度\n"
            f"v: {odom_stats['last_v']:.1f} mm/s\n"
            f"w: {math.degrees(odom_stats['last_w']):.2f} °/s\n"
            f"上次更新: {odom_stats['last_update']:.1f}s前"
        )

    def set_status(self, text: str, color: str = "black"):
        self.status_label.setText(f"状态: {text}")
        self.status_label.setStyleSheet(f"color: {color}; font-size: 16px; font-weight: bold;")

    def update_queue_status(self, current: int, maxsize: int):
        self.queue_label.setText(f"队列: {current}/{maxsize}")

    def save_map(self):
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"map_{timestamp}.png"
            self.mapper.save_map(filename)
            self.set_status(f"地图已保存: {filename}", "green")
        except Exception as e:
            self.set_status(f"保存失败: {e}", "red")

    def clear_map(self):
        self.mapper.clear_map()
        self.navigator._clear_dynamic_obstacles()
        self.navigator.cancel()
        self.navigator._align_settle_until = 0.0
        self.navigator._alignment_ack_sent = False
        self.navigator._send_alignment_ack = False
        self.navigator._align_stable_count = 0
        self.laser_odom.reset()
        self._wheel_initialized = False
        self._last_wheel_x = 0.0
        self._last_wheel_y = 0.0
        self._last_wheel_theta = 0.0
        self.send_velocity_command(0.0, 0.0, 0.0)
        self._last_vx = 0.0
        self._last_vy = 0.0
        self._last_vw = 0.0
        self._last_cmd_time = 0.0
        self._nav_was_active = False
        self._obstacle_fusion_interval = 0.5
        self._last_obstacle_fusion = 0.0
        self._alignment_ack_done = False
        self._need_replan = False
        self._pending_obstacles.clear()
        self.nav_status.setText("导航: 空闲")
        self.nav_status.setStyleSheet("font-size: 13px; color: #00ffaa;")
        self.nav_debug_label.setText("导航调试: 等待...")
        self.obstacle_label.setText("障碍物: 0个永久, 0个临时")
        self.loop_label.setText("闭环: ✗ 未闭合")
        self.loop_label.setStyleSheet("font-size: 14px; color: #ff6666; font-weight: bold;")
        self.odom_label.setText("里程计: 等待数据...")
        self.corr_label.setText("修正量: dx=0, dy=0, dθ=0°")
        self.vel_label.setText("速度: v=0 mm/s, w=0 rad/s")
        self.queue_label.setText("队列: 0/3")
        self.set_status("地图已重置为原始静态地图", "orange")

    def reset_odom(self):
        self.mapper.reset_odometry()
        self.set_status("里程计已重置", "blue")


def create_mqtt_client(frame_queue: Queue, comm: Communicate):
    client_id = f"lidar_mapper_{uuid.uuid4().hex[:8]}_{int(time.time())}"
    print(f"[MQTT] Client ID: {client_id}")

    client = mqtt.Client(client_id=client_id, clean_session=True)
    client.username_pw_set(MQTT_USER, MQTT_PASS)

    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print(f"[MQTT] 已连接到 {MQTT_HOST}:{MQTT_PORT}")
            client.subscribe(TOPIC_LIDAR, qos=0)
            print(f"[MQTT] 已订阅 {TOPIC_LIDAR}")
            comm.status_msg.emit(f"已连接 - {MQTT_HOST}", "green")
        else:
            print(f"[MQTT] 连接失败: {rc}")
            comm.status_msg.emit(f"连接失败: {rc}", "red")

    def on_message(client, userdata, msg):
        if msg.topic != TOPIC_LIDAR:
            return
        try:
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except Empty:
                    pass
            frame_queue.put_nowait(msg.payload)
        except Exception:
            pass

    def on_disconnect(client, userdata, rc):
        if rc != 0:
            print("[MQTT] 意外断开")
            comm.status_msg.emit("连接断开，正在重连...", "red")

    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect
    return client


def processing_thread(frame_queue: Queue, mapper: LidarMapper, comm: Communicate, window: MainWindow):
    drain_until = time.time() + 2.0
    frame_count = 0
    last_pose = None
    odom_valid_count = 0
    odom_invalid_count = 0

    while True:
        try:
            payload = frame_queue.get(timeout=0.5)
        except Empty:
            continue

        current_time = time.time()
        if current_time < drain_until:
            continue

        try:
            points, odom = parse_frame(payload)
            points = mapper.filter_points(points)

            frame_count += 1
            if odom.valid:
                odom_valid_count += 1
            else:
                odom_invalid_count += 1

            if frame_count % 10 == 0:
                print(f"\n[DIAG] 总帧:{frame_count} 有效里程计:{odom_valid_count} 无效:{odom_invalid_count}")
                print(f"[DIAG] 本帧: 雷达点={len(points)} payload={len(payload)}B")

            laser_success = False
            if len(points) >= 10:
                local_pts = []
                for pt in points:
                    rad = math.radians(pt.angle)
                    lx = pt.distance * math.cos(rad)
                    ly = -pt.distance * math.sin(rad)
                    local_pts.append((lx, ly))

                if odom.valid:
                    if not window._wheel_initialized:
                        window._last_wheel_x = odom.x
                        window._last_wheel_y = odom.y
                        window._last_wheel_theta = odom.theta
                        window._wheel_initialized = True

                    dx_wheel = odom.x - window._last_wheel_x
                    dy_wheel = odom.y - window._last_wheel_y
                    dtheta_wheel = odom.theta - window._last_wheel_theta

                    while dtheta_wheel > math.pi:
                        dtheta_wheel -= 2 * math.pi
                    while dtheta_wheel < -math.pi:
                        dtheta_wheel += 2 * math.pi

                    window._last_wheel_x = odom.x
                    window._last_wheel_y = odom.y
                    window._last_wheel_theta = odom.theta

                    window.laser_odom.update_wheel_odom(odom.x, odom.y, odom.theta)
                    window.laser_odom._motion_update(dx_wheel, dy_wheel, dtheta_wheel)

                success, lx, ly, ltheta = window.laser_odom.update(
                    local_pts,
                    timestamp=current_time
                )

                if success:
                    laser_success = True

                    class FakeOdom:
                        pass
                    fake_odom = FakeOdom()
                    fake_odom.x = lx
                    fake_odom.y = ly
                    fake_odom.theta = ltheta
                    fake_odom.vx = odom.vx if odom.valid else 0.0
                    fake_odom.vy = odom.vy if odom.valid else 0.0
                    fake_odom.wz = odom.wz if odom.valid else 0.0
                    fake_odom.valid = True

                    mapper.update_odom_direct(fake_odom, int(current_time * 1000))
                    last_pose = (lx, ly, ltheta)
                    comm.new_odom.emit()

                    if frame_count % 50 == 0:
                        print(f"[PF] 帧#{frame_count} "
                              f"位姿:({lx:.0f},{ly:.0f}) "
                              f"θ:{math.degrees(ltheta):.1f}° "
                              f"静止:{window.laser_odom.is_stationary}")

            if not laser_success and odom.valid:
                mapper.update_odom_direct(odom, int(current_time * 1000))
                last_pose = (odom.x, odom.y, odom.theta)
                comm.new_odom.emit()

            if len(points) >= 3:
                with mapper.lock:
                    mapper.frame_count += 1
                    mapper.total_points += len(points)

                    local_pts = []
                    world_pts = []
                    for pt in points:
                        rad = math.radians(pt.angle)
                        lx = pt.distance * math.cos(rad)
                        ly = -pt.distance * math.sin(rad)
                        local_pts.append((lx, ly))

                        wx, wy = mapper.pose.transform_point(lx, ly)
                        world_pts.append((wx, wy))
                        mapper.all_scanned_points.append((wx, wy))

                        if not mapper.static_map_mode:
                            mapper.map.update_ray(mapper.pose.x, mapper.pose.y, wx, wy)

                    mapper.latest_points_local = local_pts
                    mapper.latest_points_world = world_pts

                    if current_time - window._last_obstacle_fusion > window._obstacle_fusion_interval:
                        obstacle_candidates = []
                        robot_x = mapper.pose.x
                        robot_y = mapper.pose.y
                        for wx, wy in world_pts:
                            dx = wx - robot_x
                            dy = wy - robot_y
                            dist = math.hypot(dx, dy)
                            if 80 < dist < 2500:
                                obstacle_candidates.append((wx, wy))

                        if obstacle_candidates:
                            comm.obstacle_fusion.emit(obstacle_candidates)

                        window._last_obstacle_fusion = current_time

                comm.new_data.emit()

            window.update_queue_status(frame_queue.qsize(), FRAME_QUEUE_SIZE)

        except Exception as e:
            print(f"[ERROR] 处理帧失败: {e}")
            import traceback
            traceback.print_exc()


def main():
    warnings.filterwarnings("ignore")
    app = QApplication(sys.argv)
    font = QFont("Microsoft YaHei", 11)
    app.setFont(font)

    mapper = LidarMapper(
        map_size_mm=MAP_SIZE_MM,
        resolution_mm=RESOLUTION_MM,
        load_map_path=MAP_FILE
    )
    comm = Communicate()

    frame_queue = Queue(maxsize=FRAME_QUEUE_SIZE)
    client = create_mqtt_client(frame_queue, comm)

    window = MainWindow(mapper, comm, client)
    window.show()

    proc_thread = threading.Thread(
        target=processing_thread,
        args=(frame_queue, mapper, comm, window),
        daemon=True
    )
    proc_thread.start()

    def mqtt_thread():
        while True:
            try:
                client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
                client.loop_forever()
            except Exception as e:
                print(f"[ERROR] MQTT: {e}")
                comm.status_msg.emit(f"MQTT错误: {e}", "red")
                time.sleep(3)

    mqtt_t = threading.Thread(target=mqtt_thread, daemon=True)
    mqtt_t.start()

    try:
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        pass
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()