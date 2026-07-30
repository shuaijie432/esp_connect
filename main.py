


"""Main: 静态地图导航 + 速度闭环控制下发 - 最终修复版（适配机器人半径230mm）"""

import sys
import json
import math
import time
import threading
import warnings
import uuid
from queue import Queue, Empty
from time import sleep

import paho.mqtt.client as mqtt

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSplitter, QLineEdit
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QRect
from PyQt5.QtGui import QFont

from lidar_parser import parse_frame
from mapper import LidarMapper
from map_widget import MapWidget
from navigator import Navigator
from laser_odometry import LaserOdometry
from ws_server import WebSocketServer


MQTT_HOST = "10.113.145.227"
MQTT_PORT = 1883
MQTT_USER = "esp_send"
MQTT_PASS = "00000000"
TOPIC_LIDAR = "esp/f79541/data"
TOPIC_CONTROL = "device/f79541/data"
TOPIC_OPENMV  = "openmv/nav111"
TOPIC_OPENMV_RECV = "openmv/data"      # 接收 OpenMV 发来的数据
TOPIC_VOICE_TASK = "python/task/voice"  # 订阅: Java 后端 → Python 语音任务
TOPIC_TASK_RESULT = "python/task/result"  # 发布: Python → Java 后端任务结果


MAP_SIZE_MM = 8000
RESOLUTION_MM = 10
FRAME_QUEUE_SIZE = 3
MAP_FILE = "1.png"

# WebSocket 截图裁剪区域（相对于主窗口左上角，单位：像素）
CROP_X = 600      # 从 GUI 左边缘往右多少 px 开始截
CROP_Y = 250      # 从 GUI 上边缘往下多少 px 开始截
CROP_W = 600    # 截图宽度
CROP_H = 480    # 截图高度


class Communicate(QObject):
    new_data = pyqtSignal()
    new_odom = pyqtSignal()
    status_msg = pyqtSignal(str, str)
    obstacle_fusion = pyqtSignal(list)
    java_nav_trigger = pyqtSignal()
    nav_zero_trigger = pyqtSignal()                # MQTT "0" 触发导航 → (400, -1450) @ -90°
    openmv_data = pyqtSignal(bytes)                # OpenMV 发来的原始帧数据
    voice_task = pyqtSignal(dict)                    # Java 后端: 语音任务（JSON 解析后的 dict）


class MainWindow(QMainWindow):
    def __init__(self, mapper: LidarMapper, comm: Communicate, client: mqtt.Client = None,
                 ws_server: WebSocketServer = None):
        super().__init__()
        self.mapper = mapper
        self.comm = comm
        self.client = client
        self.ws_server = ws_server
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


        self.stats_label = QLabel("帧数: 0\n点数: 0")
        self.stats_label.setStyleSheet("font-size: 13px; line-height: 1.5;")
        right_layout.addWidget(self.stats_label)

        self.odom_label = QLabel("里程计: 等待数据...")
        self.odom_label.setStyleSheet("font-size: 13px; line-height: 1.5; color: #66ccff;")
        right_layout.addWidget(self.odom_label)


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

        self.openmv_label = QLabel("OpenMV: 等待数据...")
        self.openmv_label.setStyleSheet("font-size: 12px; line-height: 1.4; color: #66ff99;")
        self.openmv_label.setWordWrap(True)
        right_layout.addWidget(self.openmv_label)

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

        self.btn_load_wp = QPushButton("📂 加载点位文件")
        self.btn_load_wp.setStyleSheet(
            "font-size: 13px; padding: 8px; background-color: #6666cc; color: white; font-weight: bold;"
        )
        self.btn_load_wp.clicked.connect(self.load_waypoint_file)
        right_layout.addWidget(self.btn_load_wp)

        self.wp_progress_label = QLabel("点位: 未加载")
        self.wp_progress_label.setStyleSheet("font-size: 12px; color: #cc99ff; line-height: 1.4;")
        right_layout.addWidget(self.wp_progress_label)

        self.nav_status = QLabel("导航: 空闲")
        self.nav_status.setStyleSheet("font-size: 13px; color: #00ffaa;")
        right_layout.addWidget(self.nav_status)

        self.nav_target_label = QLabel("导航目标: 未设置")
        self.nav_target_label.setStyleSheet("font-size: 12px; color: #ffcc00; line-height: 1.4;")
        right_layout.addWidget(self.nav_target_label)

        right_layout.addStretch()
        splitter.addWidget(right_widget)
        splitter.setSizes([1600, 320])

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_display)
        self.timer.start(300)  # 200→300ms，减少 GUI 重绘频率
        self.comm.new_odom.connect(self.update_odom_display)
        self.comm.status_msg.connect(self.set_status)
        self.comm.obstacle_fusion.connect(self._on_obstacle_fusion)
        self.comm.java_nav_trigger.connect(self._on_java_nav_trigger)
        self.comm.nav_zero_trigger.connect(self._on_nav_zero_trigger)
        self.comm.openmv_data.connect(self._on_openmv_data)

        # 地图点击 → 设置目标并开始导航
        self.map_view.map_clicked.connect(self.on_map_clicked)


        self.comm.voice_task.connect(self._on_voice_task)

        self.nav_timer = QTimer()
        self.nav_timer.timeout.connect(self.nav_step)
        self.nav_timer.start(100)

        self._last_vx = 0.0
        self._last_vy = 0.0
        self._last_vw = 0.0
        self._last_cmd_time = 0.0
        self._last_nav_state = self.navigator.state
        self._nav_was_active = False

        self._obstacle_fusion_interval = 2.0  # 降低融合频率(原0.5s→2s)，减少重规划触发
        self._last_obstacle_fusion = 0.0

        self._hold_axes = {"vx": 0.0, "vy": 0.0, "vw": 0.0}
        self._hold_timer = QTimer()
        self._hold_timer.timeout.connect(self.on_hold_tick)
        self._hold_timer.start(100)

        self._need_replan = False
        self._last_replan_time = 0.0     # 重规划冷却，避免频繁解冻地图放大漂移
        self._alignment_ack_done = False
        self._pending_obstacles = []
        self._obstacle_processing = False
        self._auto_nav_triggered = False
        self._is_java_nav = False      # MQTT "2" 触发的导航，对齐完成后不发 OpenMV
        self._is_openmv_nav = False   # OpenMV 0xA2 触发的导航，完成后发 "1" 给 OpenMV
        self._openmv_a2_count = 0     # 0xA2 接收计数：第1次→(1170,-1520)，第2次→(25,30)
        self._suppress_lidar = False  # NAV_ZERO/JAVA_NAV 完成时抑制雷达点云
        self._current_voice_task = None  # 当前正在执行的 Java 后端语音任务（dict），完成后回传结果
        self._nav_zero_active = False # 标记 NAV_ZERO 导航进行中，完成后抑制雷达
        self._java_nav_active = False # 标记 JAVA_NAV 导航进行中，完成后抑制雷达直到 OpenMV 0xA2
        self._is_pick_nav = False    # 标记采摘水果导航，完成后向 OpenMV 发 pick_complete_msg
        self._pick_complete_msg = "1"  # 采摘导航完成时发给 OpenMV 的字符串，默认 "1"

        # ---- 点位文件顺序导航 ----
        self._waypoint_list = []       # 从JSON加载的点位列表
        self._waypoint_index = 0       # 当前点位索引

        # ---- 导航目标配置（从 nav_targets.json 加载，避免硬编码坐标） ----
        self._nav_targets_config = {}
        self._load_nav_targets()

    def _on_obstacle_fusion(self, obstacle_candidates):
        self._pending_obstacles.extend(obstacle_candidates)
        if len(self._pending_obstacles) > 5000:
            self._pending_obstacles = self._pending_obstacles[-2500:]

    def _on_java_nav_trigger(self):
        """MQTT "2" 触发 → 从 nav_targets.json 加载 'java_nav' 目标"""
        print("[JAVA_NAV] MQTT '2' 触发")
        self._apply_nav_target("java_nav")

    def _on_nav_zero_trigger(self):
        """MQTT "0" 触发 → 从 nav_targets.json 加载 'nav_zero' 目标"""
        print("[NAV_ZERO] MQTT '0' 触发")
        self._apply_nav_target("nav_zero")

    # ========== Java 后端语音任务处理 ==========

    def _on_voice_task(self, task: dict):
        """接收 Java 后端发来的语音任务，按 intentType 分发处理"""
        import time as _time

        intent = task.get("intentType", "")
        task_id = task.get("taskId", "?")
        params = task.get("parameters", {})

        print(f"[VOICE_TASK] 分发任务: intentType={intent}, taskId={task_id}, params={params}")

        if intent in ("PICK_FRUIT", "INSPECT_FRUIT"):
            self._handle_pick_fruit(task)
        elif intent == "NAVIGATE":
            self._handle_navigate(task)
        elif intent == "CANCEL":
            self._handle_cancel(task)
        else:
            print(f"[VOICE_TASK] 不支持的 intentType: {intent}")
            self._publish_task_result(task_id, "FAILED",
                                      error=f"不支持的意图类型: {intent}")

    def _handle_pick_fruit(self, task: dict):
        """采摘/检查水果: 用 fruitName 匹配 nav_targets.json 中的目标"""
        params = task.get("parameters", {})
        fruit_name = params.get("fruitName", "")
        task_id = task.get("taskId", "?")

        if not fruit_name:
            print(f"[VOICE_TASK] PICK_FRUIT 缺少 fruitName 参数")
            self._publish_task_result(task_id, "FAILED", error="缺少 fruitName 参数")
            return

        # 检查 nav_targets.json 中是否存在该水果的目标
        if fruit_name not in self._nav_targets_config:
            print(f"[VOICE_TASK] 未找到目标配置: '{fruit_name}'")
            self._publish_task_result(task_id, "FAILED",
                                      error=f"未找到目标: {fruit_name}")
            return

        print(f"[VOICE_TASK] PICK_FRUIT → '{fruit_name}'")
        self._current_voice_task = task
        self._apply_nav_target(fruit_name)
        self._is_pick_nav = True  # 放在 _apply_nav_target 之后，避免被 _clear_nav_flags 清掉

    def _handle_navigate(self, task: dict):
        """导航到指定点位: 用 command 关键词匹配 nav_targets.json，或用 target 精确匹配"""
        params = task.get("parameters", {})
        task_id = task.get("taskId", "?")

        # 优先用 command 字段做关键词匹配
        command = params.get("command", "")
        target_name = params.get("target", "")

        matched = None

        if command:
            # 关键词子串匹配 nav_targets.json 的 key（按长度降序，长关键词优先）
            sorted_keys = sorted(self._nav_targets_config.keys(), key=len, reverse=True)
            for key in sorted_keys:
                if key in command:
                    matched = key
                    print(f"[VOICE_TASK] 关键词匹配: '{command}' → '{key}'")
                    break

        if not matched and target_name:
            matched = target_name

        if not matched:
            print(f"[VOICE_TASK] NAVIGATE 无法匹配目标: command='{command}', target='{target_name}'")
            self._publish_task_result(task_id, "FAILED",
                                      error="无法从指令中识别导航目标")
            return

        if matched not in self._nav_targets_config:
            print(f"[VOICE_TASK] 未找到目标配置: '{matched}'")
            self._publish_task_result(task_id, "FAILED",
                                      error=f"未找到目标: {matched}")
            return

        print(f"[VOICE_TASK] NAVIGATE → '{matched}'")
        self._current_voice_task = task
        self._apply_nav_target(matched)

    def _handle_cancel(self, task: dict):
        """取消当前导航任务"""
        task_id = task.get("taskId", "?")
        print(f"[VOICE_TASK] CANCEL → 取消当前导航")

        self._current_voice_task = None
        self.stop_navigation()
        self._publish_task_result(task_id, "COMPLETED",
                                  data={"action": "cancel"})

    def _publish_task_result(self, task_id: str, status: str,
                              data: dict = None, error: str = None):
        """向 Java 后端发布任务执行结果 (topic: python/task/result)"""
        import time as _time

        if self.client is None or not self.client.is_connected():
            print(f"[VOICE_TASK] MQTT 未连接，无法发布结果 (taskId={task_id})")
            return

        result = {
            "taskId": task_id,
            "status": status,
            "timestamp": int(_time.time() * 1000),
        }
        if data is not None:
            result["data"] = data
        if error is not None:
            result["error"] = error

        try:
            payload = json.dumps(result, ensure_ascii=False)
            self.client.publish(TOPIC_TASK_RESULT, payload, qos=1)
            print(f"[VOICE_TASK] 结果已发布 → {TOPIC_TASK_RESULT}: {payload}")
        except Exception as e:
            print(f"[VOICE_TASK] 发布结果失败: {e}")

    # ========== OpenMV 数据处理 ==========

    def _on_openmv_data(self, data: bytes):
        """处理 OpenMV 发来的数据"""
        hex_str = ' '.join(f'{b:02X}' for b in data)
        data_len = len(data)
        print(f"[OPENMV] 数据处理: {hex_str} (长度={data_len})")

        # 根据数据内容做具体处理
        if data_len == 1:
            cmd = data[0]
            if cmd == 0xA2:
                self._openmv_a2_count += 1
                if self._openmv_a2_count == 1:
                    print(f"[OPENMV] 第1次收到 0xA2 → 跳过，不导航")
                    self.openmv_label.setText(f"OpenMV: 0xA2 (#1) → 跳过")
                    self.openmv_label.setStyleSheet(
                        "font-size: 12px; line-height: 1.4; color: #ffcc00;"
                    )
                else:
                    print(f"[OPENMV] 第{self._openmv_a2_count}次收到 0xA2 → 导航至 openmv_a2 目标")
                    self.openmv_label.setText(f"OpenMV: 0xA2 (#{self._openmv_a2_count})")
                    self.openmv_label.setStyleSheet(
                        "font-size: 12px; line-height: 1.4; color: #66ff66;"
                    )
                    self._apply_nav_target("openmv_a2")
            elif cmd == 0x02:
                self.openmv_label.setText(f"OpenMV: 收到指令 0x02")
                self.openmv_label.setStyleSheet(
                    "font-size: 12px; line-height: 1.4; color: #ffcc00;"
                )
                print("[OPENMV] → 指令 0x02")
            else:
                self.openmv_label.setText(f"OpenMV: 0x{cmd:02X}")
                self.openmv_label.setStyleSheet(
                    "font-size: 12px; line-height: 1.4; color: #66ff99;"
                )
        else:
            self.openmv_label.setText(f"OpenMV: [{hex_str}]")
            self.openmv_label.setStyleSheet(
                "font-size: 12px; line-height: 1.4; color: #66ff99;"
            )

    def on_map_clicked(self, wx: float, wy: float):
        """地图点击回调：更新目标坐标输入框并自动开始导航"""
        self.target_x.setText(f"{wx:.0f}")
        self.target_y.setText(f"{wy:.0f}")
        self.nav_target_label.setText(
            f"导航目标: ({wx:.0f}, {wy:.0f}) mm\n点击地图设置新目标"
        )
        self.nav_target_label.setStyleSheet(
            "font-size: 12px; color: #00ffaa; line-height: 1.4;"
        )
        self.start_navigation()

    # ============================================================
    # 点位文件加载 & 顺序导航
    # ============================================================
    def load_waypoint_file(self):
        """加载点位JSON文件并启动顺序导航"""
        from PyQt5.QtWidgets import QFileDialog
        import json

        filepath, _ = QFileDialog.getOpenFileName(
            self, "选择点位文件", "", "JSON Files (*.json)"
        )
        if not filepath:
            return

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            self.set_status(f"点位文件加载失败: {e}", "red")
            return

        if not isinstance(data, list) or len(data) == 0:
            self.set_status("点位文件格式错误（需要非空数组）", "red")
            return

        self._waypoint_list = data
        self._waypoint_index = 0



        self.set_status(
            f"点位文件已加载: {len(data)}个目标点 | {filepath.split('/')[-1]}",
            "green"
        )
        self._start_waypoint(0)

    # ========== 导航目标配置加载 ==========

    def _load_nav_targets(self):
        """从 nav_targets.json 加载所有导航目标配置"""
        import json
        import os

        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nav_targets.json")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._nav_targets_config = data.get("targets", {})
            print(f"[CONFIG] 已加载 {len(self._nav_targets_config)} 个导航目标: "
                  f"{list(self._nav_targets_config.keys())}")
        except FileNotFoundError:
            print(f"[CONFIG] 未找到 nav_targets.json，将使用代码中的默认值")
            self._nav_targets_config = {}
        except Exception as e:
            print(f"[CONFIG] 加载 nav_targets.json 失败: {e}")
            self._nav_targets_config = {}

    def _clear_nav_flags(self):
        """清除所有导航状态标记，防止上一次导航的遗留标记干扰新导航"""
        self._is_java_nav = False
        self._is_openmv_nav = False
        self._java_nav_active = False
        self._nav_zero_active = False
        self._is_pick_nav = False
        self._pick_complete_msg = "1"

    def _apply_nav_target(self, name: str):
        """从配置中加载指定目标的参数并启动导航

        配置文件 nav_targets.json 中每个 target 可包含:
            - x, y, theta_deg         (必填) 目标坐标与角度
            - final_approach_dist     (可选) 最终接近距离
            - safety_boost            (可选) 碰撞框额外扩大
            - is_java_nav / is_openmv_nav (可选) 导航来源标记
            - reset_nav_count         (可选) 重置导航计数
            - flags                   (可选) 需要设置的标记位字典
        """
        target = self._nav_targets_config.get(name)
        if not target:
            print(f"[NAV_TARGET] 未找到目标配置 '{name}'，回退到默认值")
            return False

        # 清除上一次导航的遗留标记，避免多个完成处理器同时触发
        self._clear_nav_flags()

        desc = target.get("description", name)
        tx = target["x"]
        ty = target["y"]
        tdeg = target["theta_deg"]

        print(f"[NAV_TARGET] {desc} → ({tx}, {ty}) @ {tdeg}°")

        # 设置目标坐标
        self.target_x.setText(str(tx))
        self.target_y.setText(str(ty))
        self.target_theta_deg.setText(str(tdeg))

        # 可选: 导航器参数
        if "final_approach_dist" in target:
            self.navigator.final_approach_dist = float(target["final_approach_dist"])
        if "safety_boost" in target:
            self.navigator.safety_boost = float(target["safety_boost"])

        # 可选: 导航来源标记
        if "is_java_nav" in target:
            self._is_java_nav = bool(target["is_java_nav"])
        if "is_openmv_nav" in target:
            self._is_openmv_nav = bool(target["is_openmv_nav"])

        # 可选: 重置导航计数（模拟首次导航走起点保护）
        if target.get("reset_nav_count"):
            self.navigator._nav_count = 0

        # 通用 flags
        flags = target.get("flags", {})
        for flag_name, flag_value in flags.items():
            attr_name = f"_{flag_name}"
            if hasattr(self, attr_name):
                setattr(self, attr_name, bool(flag_value))
            else:
                print(f"[NAV_TARGET] 警告: 未知标记 '{flag_name}'，已跳过")

        # 可选: 采摘完成时发给 OpenMV 的字符串（默认 "1"）
        if "pick_complete_msg" in target:
            self._pick_complete_msg = str(target["pick_complete_msg"])
            print(f"[NAV_TARGET] pick_complete_msg = '{self._pick_complete_msg}'")

        self.start_navigation()
        return True

    def _start_waypoint(self, index: int):
        """启动第 index 个点位的导航"""
        if index >= len(self._waypoint_list):
            # 所有点位已完成
            self.wp_progress_label.setText(
                f"点位: ✅ 全部完成 ({len(self._waypoint_list)}/{len(self._waypoint_list)})"
            )
            self.wp_progress_label.setStyleSheet(
                "font-size: 12px; color: #66ff66; line-height: 1.4;"
            )
            self.set_status("所有点位导航已完成！", "green")
            self.btn_load_wp.setText("📂 加载点位文件")
            return

        wp = self._waypoint_list[index]
        name = wp.get("name", f"点{index+1}")
        tx = float(wp["x"])
        ty = float(wp["y"])
        ttheta_deg = float(wp.get("theta_deg", 0))

        # 更新界面输入框
        self.target_x.setText(f"{tx:.0f}")
        self.target_y.setText(f"{ty:.0f}")
        self.target_theta_deg.setText(f"{ttheta_deg:.0f}")

        # 更新点位进度标签
        self.wp_progress_label.setText(
            f"点位: [{index+1}/{len(self._waypoint_list)}] {name}"
        )
        self.wp_progress_label.setStyleSheet(
            "font-size: 12px; color: #cc99ff; line-height: 1.4;"
        )
        self.btn_load_wp.setText(f"📍 [{index+1}/{len(self._waypoint_list)}] {name}")

        print(f"[WP] 启动点位 [{index+1}/{len(self._waypoint_list)}] "
              f"\"{name}\" → ({tx:.0f}, {ty:.0f}) @{ttheta_deg:.0f}°")

        self.start_navigation()

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
            # 任何障碍物变化都触发重规划：新增、位置变化、新固化为永久
            # 但 ALIGNING（对准角度）期间不触发，避免打断角度对准
            obstacle_changed = (result.get('updated', 0) > 0 or
                                result.get('newly_permanent', 0) > 0 or
                                result.get('added', 0) > 0)
            if obstacle_changed and self.navigator.is_active() and self.navigator.state != "ALIGNING":
                print(f"[MAP_FUSION] 障碍物变化(新增={result.get('updated',0)} "
                      f"固化={result.get('newly_permanent',0)} 写入={result.get('added',0)})，触发重规划")
                self._need_replan = True
        except Exception as e:
            print(f"[ERROR] 障碍物融合失败: {e}")
            import traceback
            traceback.print_exc()

    def start_navigation(self):
        # 开始新导航时，重新开启雷达数据接收
        self._suppress_lidar = False

        try:
            tx = float(self.target_x.text())
            ty = float(self.target_y.text())
            ttheta_deg = float(self.target_theta_deg.text())
            ttheta_rad = math.radians(ttheta_deg)

            success = self.navigator.set_target(tx, ty, ttheta_rad)

            if success:
                self.set_status(f"开始导航 → ({tx:.0f}, {ty:.0f}) @{ttheta_deg:.0f}°", "green")
                self.nav_status.setText(self.navigator.get_status())
                self.nav_target_label.setText(
                    f"导航目标: ({tx:.0f}, {ty:.0f}) mm\n"
                    f"角度: {ttheta_deg:.0f}° | 路径点: {len(self.navigator.get_waypoints())}个"
                )
                self.nav_target_label.setStyleSheet(
                    "font-size: 12px; color: #00ffaa; line-height: 1.4;"
                )
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

        def _send_openmv():
            if self.client is not None and self.client.is_connected():
                try:
                    msg = self._pick_complete_msg
                    self.client.publish(TOPIC_OPENMV, msg, qos=1)
                    print(f"[CMD] 角度对准标志位已发送(OpenMV) -> {TOPIC_OPENMV} = '{msg}'")
                except Exception as e:
                    print(f"[CMD] OpenMV发送失败: {e}")

        _send_openmv()

        self._alignment_ack_done = True

    def _send_openmv_signal(self):
        """点1到达后，发 pick_complete_msg 给 OpenMV"""
        if self.client is None or not self.client.is_connected():
            print("[WP] MQTT未连接，无法发送OpenMV信号")
            return
        try:
            msg = self._pick_complete_msg
            self.client.publish(TOPIC_OPENMV, msg, qos=1)
            print(f"[WP] 点1到达 → OpenMV信号已发送 -> {TOPIC_OPENMV} = '{msg}'")
        except Exception as e:
            print(f"[WP] OpenMV信号发送失败: {e}")

    def send_last_waypoint_frame(self):
        """到达最后点位后，发送 0xA2 帧到 ESP32"""
        if self.client is None or not self.client.is_connected():
            print("[WP] MQTT未连接，无法发送最后点位完成帧")
            return

        frame = bytearray()
        frame.append(0xCC)
        frame.append(0x66)
        frame.append(0x01)
        frame.append(0xA2)
        checksum = sum(frame[2:]) & 0xFF
        frame.append(checksum)
        frame.append(0xEE)

        try:
            self.client.publish(TOPIC_CONTROL, bytes(frame), qos=1)
            hex_str = ' '.join(f'{b:02X}' for b in frame)
            print(f"[WP] 最后点位完成帧已发送 -> {TOPIC_CONTROL} | HEX[{hex_str}]")
        except Exception as e:
            print(f"[WP] 最后点位完成帧发送失败: {e}")

    def on_manual_send_ack(self):
        self.send_last_waypoint_frame()
        self._alignment_ack_done = True
        self.navigator.cancel()
        self.navigator._align_settle_until = 0.0
        self.navigator._alignment_ack_sent = False
        self.navigator._send_alignment_ack = False
        self.navigator._align_stable_count = 0
        self.nav_status.setText("导航: 手动发送完成帧")
        self.nav_status.setStyleSheet("font-size: 13px; color: #3366aa;")
        self.set_status("已手动发送 0xA2 完成帧，速度下发已停止", "blue")

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
        self.nav_target_label.setText("导航目标: 未设置")
        self.nav_target_label.setStyleSheet("font-size: 12px; color: #ffcc00; line-height: 1.4;")
        # 重置点位顺序导航
        self._waypoint_list.clear()
        self._waypoint_index = 0
        self.wp_progress_label.setText("点位: 未加载")
        self.wp_progress_label.setStyleSheet("font-size: 12px; color: #cc99ff; line-height: 1.4;")
        self.btn_load_wp.setText("📂 加载点位文件")
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

                # ---- 导航到达目标点 → 屏蔽雷达数据，下次启动导航时通过 flags 重新开启 ----
                if self.navigator.state == "DONE":
                    self._suppress_lidar = True
                    with self.mapper.lock:
                        self.mapper.latest_points_local = []
                        self.mapper.latest_points_world = []
                    print(f"[LIDAR] 导航到达目标点，屏蔽雷达数据")

                # ---- 语音任务导航完成 → 向 Java 后端发布结果 ----
                if self._current_voice_task and self.navigator.state == "DONE":
                    task = self._current_voice_task
                    self._current_voice_task = None
                    self._is_pick_nav = False
                    task_id = task.get("taskId", "?")
                    intent = task.get("intentType", "")
                    params = task.get("parameters", {})
                    fruit_name = params.get("fruitName", params.get("target", ""))

                    # 构造返回数据
                    result_data = {
                        "action": "pick" if intent == "PICK_FRUIT" else "navigate",
                        "fruit": fruit_name,
                        "position": 2,       # TODO: 硬件反馈的实际位置
                        "errorX": 15,          # TODO: 硬件反馈的实际误差
                    }
                    self._publish_task_result(task_id, "COMPLETED", data=result_data)

                # ---- 点位顺序导航：当前点完成 → 自动启动下一点 ----
                if (self.navigator.state == "DONE"
                        and self._waypoint_list
                        and self._waypoint_index + 1 < len(self._waypoint_list)):

                    # 点1 完成 → 仅采摘导航时发 "1" 给 OpenMV
                    if self._waypoint_index == 0 and self._is_pick_nav:
                        self._send_openmv_signal()

                    self._waypoint_index += 1
                    # 延迟一帧再启动，确保状态已完全清理
                    QTimer.singleShot(500, lambda idx=self._waypoint_index: self._start_waypoint(idx))

                # ---- 最后点位完成 → 发送 0xA2 帧到 ESP32 ----
                elif (self.navigator.state == "DONE"
                        and self._waypoint_list
                        and self._waypoint_index + 1 >= len(self._waypoint_list)):
                    self.send_last_waypoint_frame()

                # ---- MQTT "2" 导航完成 → 发送 0xA2 帧到 ESP32 ----
                elif self.navigator.state == "DONE" and self._is_java_nav:
                    self.send_last_waypoint_frame()
                    self._is_java_nav = False

                # ---- MQTT "0" (NAV_ZERO) 导航完成 → 抑制雷达点云 ----
                if self._nav_zero_active and self.navigator.state == "DONE":
                    self._suppress_lidar = True
                    self._nav_zero_active = False
                    with self.mapper.lock:
                        self.mapper.latest_points_local = []
                        self.mapper.latest_points_world = []
                    print("[LIDAR] NAV_ZERO 导航完成，开始抑制雷达点云接收与绘制")

                # ---- JAVA_NAV 导航完成 → 抑制雷达点云，等待 OpenMV 0xA2 ----
                if self._java_nav_active and self.navigator.state == "DONE":
                    self._suppress_lidar = True
                    self._java_nav_active = False
                    with self.mapper.lock:
                        self.mapper.latest_points_local = []
                        self.mapper.latest_points_world = []
                    print("[LIDAR] JAVA_NAV 导航完成，开始抑制雷达点云接收与绘制（等待 OpenMV 0xA2 触发）")

                # ---- OpenMV 0xA2 导航完成 ----
                elif self.navigator.state == "DONE" and self._is_openmv_nav:
                    self._is_openmv_nav = False
            return

        self._nav_was_active = True

        with self.mapper.lock:
            x, y, theta = self.mapper.pose.x, self.mapper.pose.y, self.mapper.pose.theta

        # ---- 定期检测路径是否被新障碍物阻挡（每 0.5 秒） ----
        now = time.time()
        if not hasattr(self, '_last_path_blocked_check'):
            self._last_path_blocked_check = 0.0
        if (now - self._last_path_blocked_check > 2.0
                and self.navigator.state == "FOLLOWING"
                and not self._need_replan):
            self._last_path_blocked_check = now
            if self.navigator.check_path_blocked():
                print("[PATH_CHECK] 检测到前方路径点被障碍物阻挡，触发重规划")
                self._need_replan = True

        if self._need_replan:
            self._need_replan = False
            # 冷却时间：5秒内不重复重规划，避免 A* 阻塞主线程导致 DWA 和雷达处理延迟
            if now - self._last_replan_time < 1.0:
                pass  # 跳过，等冷却
            elif self.navigator.waypoints:
                target = self.navigator.waypoints[-1]
                target_theta = self.navigator.target_theta
                old_state = self.navigator.state
                old_wp = self.navigator.current_wp

                success = self.navigator.set_target(target[0], target[1], target_theta)
                if success:
                    self._last_replan_time = now
                    rx, ry = self.mapper.pose.x, self.mapper.pose.y
                    best_idx = 0
                    best_dist = float('inf')
                    for i, (wx, wy) in enumerate(self.navigator.waypoints):
                        d = math.hypot(wx - rx, wy - ry)
                        if d < best_dist:
                            best_dist = d
                            best_idx = i
                    self.navigator.current_wp = best_idx
                    self.set_status("检测到新障碍物，已重新规划路径", "orange")
                    print(f"[REPLAN] 机器人位置({rx:.0f},{ry:.0f})，最近新路径点 {best_idx}/{len(self.navigator.waypoints)} (距离{best_dist:.0f}mm)")
                else:
                    self.set_status("重规划失败！目标不可达", "red")
                    self.navigator.state = old_state

        vx, vy, vw = self.navigator.update(x, y, theta)
        status_text = self.navigator.get_status()
        wp = self.navigator.get_waypoints()
        curr = self.navigator.current_wp

        if send_ack:
            # 只有采摘水果导航才向 OpenMV 发 "1"
            if self._is_pick_nav:
                self.send_alignment_ack_frame()
                print(f"[NAV] 采摘导航角度对准完成，已向OpenMV发'1'，最终角度: {math.degrees(theta):.1f}°")
            else:
                print(f"[NAV] 非采摘导航角度对准完成，跳过OpenMV信号")
            self.navigator.state = "DONE"

        now = time.time()
        # 检测导航状态切换：状态变化时强制发送速度指令，避免限流器导致
        # ESP32 继续执行旧指令（例如 FOLLOWING→ALIGNING 时侧移未清零）。
        nav_state = self.navigator.state
        state_changed = nav_state != self._last_nav_state
        self._last_nav_state = nav_state

        if state_changed or \
           (now - self._last_cmd_time) > 0.1 or \
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

        # 合速度不低于80mm/s（合速度不为零但低于阈值时，按比例放大）
        speed = math.hypot(vx, vy)
        if speed > 0.01 and speed < 80.0:
            scale = 80.0 / speed
            vx *= scale
            vy *= scale

        try:
            import struct
            vx_m = vx / 1000.0
            vy_m = vy / 1000.0
            data = struct.pack('<fff', vx_m, -vy_m, -vw)
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
            if self._cmd_cnt % 5 == 0:  # 每5帧(0.5s)打印，方便观察DWA避障
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

        # WebSocket 推送 GUI 截图到前端（裁剪：从窗口左边缘 x px，上边缘 y px，宽 w px，高 h px）
        if self.ws_server:
            crop_rect = QRect(CROP_X, CROP_Y, CROP_W, CROP_H)
            self.ws_server.broadcast_window_image(quality=65, crop_rect=crop_rect)

    def update_odom_display(self):
        odom_stats = self.mapper.get_odom_stats()
        self.odom_label.setText(
            f"里程计积分\n"
            f"X: {odom_stats['x']:.1f}mm\n"
            f"Y: {odom_stats['y']:.1f}mm\n"
            f"θ: {math.degrees(odom_stats['theta']):.2f}°\n"
            f"轨迹点数: {odom_stats['trajectory_len']}"
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
        # 1. 重置地图数据
        self.mapper.clear_map()

        # 2. 重新创建 Navigator（跟刚打开程序一样）
        self.navigator = Navigator(self.mapper)
        self.map_view.navigator = self.navigator

        # 3. 重置激光里程计
        self.laser_odom.reset()
        if self.mapper.static_map_mode and self.mapper.map is not None:
            self.laser_odom.set_map(self.mapper.map)

        # 4. 重置轮式里程计状态
        self._wheel_initialized = False
        self._last_wheel_x = 0.0
        self._last_wheel_y = 0.0
        self._last_wheel_theta = 0.0

        # 5. 停止并重置速度下发
        self.send_velocity_command(0.0, 0.0, 0.0)
        self._last_vx = 0.0
        self._last_vy = 0.0
        self._last_vw = 0.0
        self._last_cmd_time = 0.0
        self._last_nav_state = self.navigator.state
        self._nav_was_active = False

        # 6. 重置障碍物融合状态
        self._obstacle_fusion_interval = 2.0
        self._last_obstacle_fusion = 0.0
        self._alignment_ack_done = False
        self._need_replan = False
        self._auto_nav_triggered = False
        self._pending_obstacles.clear()

        # 7. 重置所有界面标签
        self.nav_status.setText("导航: 空闲")
        self.nav_status.setStyleSheet("font-size: 13px; color: #00ffaa;")
        self.nav_target_label.setText("导航目标: 未设置")
        self.nav_target_label.setStyleSheet("font-size: 12px; color: #ffcc00; line-height: 1.4;")
        self.nav_debug_label.setText("导航调试: 等待...")
        self.obstacle_label.setText("障碍物: 0个永久, 0个临时")
        self.openmv_label.setText("OpenMV: 等待数据...")
        self.openmv_label.setStyleSheet("font-size: 12px; line-height: 1.4; color: #66ff99;")
        self.odom_label.setText("里程计: 等待数据...")

        self.vel_label.setText("速度: v=0 mm/s, w=0 rad/s")
        self.queue_label.setText("队列: 0/3")

        print("[MAIN] 程序已完全重置到初始状态")

    def reset_odom(self):
        self.mapper.reset_odometry()
        self.set_status("里程计已重置", "blue")


def create_mqtt_client(frame_queue: Queue, comm: Communicate):
    import json

    client_id = f"lidar_mapper_{uuid.uuid4().hex[:8]}_{int(time.time())}"
    print(f"[MQTT] Client ID: {client_id}")

    client = mqtt.Client(client_id=client_id, clean_session=True)
    client.username_pw_set(MQTT_USER, MQTT_PASS)

    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print(f"[MQTT] 已连接到 {MQTT_HOST}:{MQTT_PORT}")
            client.subscribe(TOPIC_LIDAR, qos=0)
            client.subscribe(TOPIC_OPENMV_RECV, qos=0)
            client.subscribe(TOPIC_VOICE_TASK, qos=1)
            print(f"[MQTT] 已订阅 {TOPIC_LIDAR}")
            print(f"[MQTT] 已订阅 {TOPIC_OPENMV_RECV} (OpenMV接收)")
            print(f"[MQTT] 已订阅 {TOPIC_VOICE_TASK} (Java语音任务)")
            comm.status_msg.emit(f"已连接 - {MQTT_HOST}", "green")
        else:
            print(f"[MQTT] 连接失败: {rc}")
            comm.status_msg.emit(f"连接失败: {rc}", "red")

    def on_message(client, userdata, msg):
        # ---- Java 后端语音任务 ----
        if msg.topic == TOPIC_VOICE_TASK:
            try:
                payload_str = msg.payload.decode('utf-8') if isinstance(msg.payload, bytes) else msg.payload
                task = json.loads(payload_str)
                intent = task.get("intentType", "UNKNOWN")
                task_id = task.get("taskId", "?")
                print(f"[VOICE_TASK] 收到任务: {intent} (taskId={task_id})")
                comm.voice_task.emit(task)
            except Exception as e:
                print(f"[VOICE_TASK] JSON 解析失败: {e}")
            return

        # ---- 激光雷达数据 ----
        if msg.topic == TOPIC_LIDAR:
            if msg.payload == b"2":
                print(f"[NAV_TRIGGER] 收到导航触发标志 -> 导航至 (1285, -145) @ -90°")
                comm.java_nav_trigger.emit()
                return
            if msg.payload == b"0":
                print(f"[NAV_TRIGGER] 收到 '0' -> 导航至 (400, -1450) @ -90°")
                comm.nav_zero_trigger.emit()
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
            return

        # ---- OpenMV 接收数据 (直接接收，兼容二进制 + ASCII hex 文本) ----
        if msg.topic == TOPIC_OPENMV_RECV:
            payload = msg.payload

            # 尝试转为纯二进制字节
            raw_bytes = None
            if isinstance(payload, (bytes, bytearray)):
                # 方式1: 已经是二进制字节
                if len(payload) <= 64 and all(b < 0x7F for b in payload):
                    # 可能是 ASCII 文本，尝试解码
                    try:
                        text = payload.decode('ascii').strip()
                        hex_str = ''.join(text.split())
                        if all(c in '0123456789abcdefABCDEF' for c in hex_str):
                            raw_bytes = bytes.fromhex(hex_str)
                    except (ValueError, UnicodeDecodeError):
                        pass
                if raw_bytes is None:
                    raw_bytes = bytes(payload)

            if raw_bytes is None:
                return

            hex_str = ' '.join(f'{b:02X}' for b in raw_bytes)
            print(f"[OPENMV] 收到: {hex_str}")
            comm.openmv_data.emit(raw_bytes)
            return

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
            payload = frame_queue.get(timeout=0.1)
        except Empty:
            time.sleep(0.01)  # 空闲时释放 GIL
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
            if not window._suppress_lidar and len(points) >= 10:
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

                # 将真实里程计增量传入 update()，避免双重加噪
                success, lx, ly, ltheta = window.laser_odom.update(
                    local_pts,
                    timestamp=current_time,
                    dx_wheel=dx_wheel if odom.valid else 0.0,
                    dy_wheel=dy_wheel if odom.valid else 0.0,
                    dtheta_wheel=dtheta_wheel if odom.valid else 0.0
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

            if not window._suppress_lidar and len(points) >= 3:
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

            time.sleep(0.001)  # 每帧释放 GIL，防止卡 GUI

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

    # 启动 WebSocket 服务（推送 GUI 截图到前端）
    ws_server = WebSocketServer(host="0.0.0.0", port=8765)
    ws_server.start()

    frame_queue = Queue(maxsize=FRAME_QUEUE_SIZE)
    client = create_mqtt_client(frame_queue, comm)

    window = MainWindow(mapper, comm, client, ws_server)
    ws_server.set_main_window(window)
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