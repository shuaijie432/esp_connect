"""
WebSocket 服务：将 PyQt5 GUI 界面截图实时推送到 Vue 前端
使用 FastAPI + asyncio，在独立线程中运行，不阻塞 PyQt5 主循环

- 截图缩小到 960px 宽，减少传输量
- 支持多客户端同时连接
"""

import json
import asyncio
import threading
import time
import base64
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from PyQt5.QtWidgets import QMainWindow, QWidget
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QBuffer, QIODevice, QPoint, QRect

MAX_CLIENTS = 20  # 允许多台设备同时查看
MAX_IMAGE_WIDTH = 2560  # 高清晰度，适配 2K 屏幕


def _qimage_to_jpeg_bytes(qimage: QImage, quality: int = 80) -> bytes:
    """QImage → JPEG 字节流"""
    buf = QBuffer()
    buf.open(QIODevice.WriteOnly)
    qimage.save(buf, "JPEG", quality=quality)
    return bytes(buf.data())


class WebSocketServer:
    """WebSocket 服务器，管理所有连接的客户端并广播截图"""

    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.port = port
        self._clients: set[WebSocket] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._main_window: Optional[QMainWindow] = None
        self._map_view: Optional[QWidget] = None  # MapWidget，用于坐标转换
        self._client_counter = 0  # 用于日志追踪

        # ---- FastAPI app ----
        self._app = FastAPI()
        self._app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @self._app.websocket("/ws")
        async def endpoint(ws: WebSocket):
            await ws.accept()

            # 超过限制：踢掉最旧的（fire-and-forget close，不 await！）
            # await 会 yield 让其他协程插入，导致两个连接同时绕过检查
            while len(self._clients) >= MAX_CLIENTS:
                old = next(iter(self._clients))
                self._clients.discard(old)
                # 不 await，直接创建后台任务关闭
                asyncio.create_task(self._safe_close(old))

            self._client_counter += 1
            cid = self._client_counter
            self._clients.add(ws)
            print(f"[WS] #{cid} 已连接 (共 {len(self._clients)} 个)")

            try:
                while True:
                    data = await ws.receive_text()
                    await self._handle_client_message(ws, data)
            except WebSocketDisconnect:
                pass
            except Exception:
                pass
            finally:
                self._clients.discard(ws)
                print(f"[WS] #{cid} 已断开 (共 {len(self._clients)} 个)")

    # ==================== 绑定主窗口 ====================

    def set_main_window(self, window: QMainWindow):
        self._main_window = window

    def set_map_view(self, widget: QWidget):
        """绑定 MapWidget，用于前端点击时转换坐标"""
        self._map_view = widget

    # ==================== 广播方法（从 PyQt5 主线程调用） ====================

    def broadcast_full_state(self):
        """推送完整状态数据：位姿、地图、点云、轨迹、导航路径"""
        if not self._main_window or not self._clients:
            return
        mapper = self._main_window.mapper
        navigator = self._main_window.navigator
        stats = mapper.get_stats()
        odom = mapper.get_odom_stats()

        # 地图栅格（降采样到 ~200x200）
        map_grid = None
        map_size_mm = 8000
        resolution = 15
        if mapper.map is not None:
            grid = mapper.map.get_display()
            step = max(1, grid.shape[0] // 200)
            map_grid = grid[::step, ::step].tolist()
            map_size_mm = mapper.map.size
            resolution = mapper.map.resolution * step

        # 激光点云（世界坐标，最近 500 个）
        _, world_pts = mapper.get_latest_points()
        scan_points = [{"x": p[0], "y": p[1]} for p in world_pts[-500:]]

        # 轨迹（最近 1000 个）
        traj = [{"x": t[0], "y": t[1]} for t in mapper.get_trajectory()[-1000:]]

        # 导航路径
        nav_path = []
        if navigator and navigator.is_active():
            pts = navigator.get_waypoints()
            if pts:
                nav_path = [{"x": p[0], "y": p[1]} for p in pts]

        data = {
            "type": "full_state", "timestamp": time.time(),
            "pose": {"x": stats["pose"][0], "y": stats["pose"][1], "theta": stats["pose"][2]},
            "stats": {
                "frame_count": stats["frame_count"], "total_points": stats["total_points"],
                "history_points": stats["history_points"], "map_size": stats["map_size"],
                "loop_detected": stats["loop_detected"],
            },
            "odom": {
                "x": odom["x"], "y": odom["y"], "theta": odom["theta"],
                "trajectory_len": odom["trajectory_len"],
                "last_v": odom["last_v"], "last_w": odom["last_w"],
                "last_update": odom["last_update"],
                "loop_correction": list(odom["loop_correction"]),
            },
            "map_grid": map_grid, "map_size_mm": map_size_mm, "map_resolution": resolution,
            "scan_points": scan_points, "trajectory": traj, "nav_path": nav_path,
            "nav_status": navigator.get_status() if navigator else "空闲",
            "nav_active": navigator.is_active() if navigator else False,
        }
        asyncio.run_coroutine_threadsafe(
            self._broadcast_text(json.dumps(data)), self._loop
        )

    def broadcast_window_image(self, quality: int = 95, crop_rect: Optional[QRect] = None):
        """截取主窗口 → 裁剪(可选) → 缩放 → JPEG → base64 → JSON 广播"""
        if not self._main_window or not self._clients:
            return

        # 截图
        pixmap = self._main_window.grab()
        qimage = pixmap.toImage().convertToFormat(QImage.Format_RGB888)

        # 裁剪（只保留 crop_rect 区域）
        if crop_rect is not None and crop_rect.isValid():
            # 确保裁剪区域在图像范围内
            x = max(0, crop_rect.x())
            y = max(0, crop_rect.y())
            w = min(crop_rect.width(), qimage.width() - x)
            h = min(crop_rect.height(), qimage.height() - y)
            qimage = qimage.copy(x, y, w, h)

        # 缩放（宽 > MAX_IMAGE_WIDTH 时等比缩小）
        if qimage.width() > MAX_IMAGE_WIDTH:
            ratio = MAX_IMAGE_WIDTH / qimage.width()
            new_h = int(qimage.height() * ratio)
            qimage = qimage.scaled(
                MAX_IMAGE_WIDTH, new_h,
                aspectRatioMode=1,  # Qt.KeepAspectRatio
                transformMode=1     # Qt.SmoothTransformation
            )

        # JPEG 压缩 + base64 编码
        jpeg_bytes = _qimage_to_jpeg_bytes(qimage, quality=quality)
        b64 = base64.b64encode(jpeg_bytes).decode()

        data = {
            "type": "window_image",
            "width": qimage.width(),
            "height": qimage.height(),
            "image": b64,
            "format": "jpeg",
            "timestamp": time.time(),
        }
        asyncio.run_coroutine_threadsafe(
            self._broadcast_text(json.dumps(data)), self._loop
        )

    # ==================== 内部 ====================

    @staticmethod
    async def _safe_close(ws: WebSocket):
        """安全关闭连接，忽略所有异常"""
        try:
            await ws.close()
        except Exception:
            pass

    async def _broadcast_text(self, text: str):
        """文本广播（带超时保护，自动清理死连接）"""
        dead = []
        for ws in list(self._clients):
            try:
                await asyncio.wait_for(ws.send_text(text), timeout=5.0)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)
            try:
                await ws.close()
            except Exception:
                pass
        if dead:
            print(f"[WS] 清理 {len(dead)} 个死连接 (共 {len(self._clients)} 个)")

    async def _handle_client_message(self, ws: WebSocket, raw: str):
        """处理前端发来的消息"""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        action = msg.get("action", "")
        if action == "map_click":
            # 前端点击地图：img_x/img_y 是图片上的像素坐标，img_w/img_h 是图片尺寸
            if self._on_map_click:
                self._on_map_click(
                    msg.get("img_x", 0), msg.get("img_y", 0),
                    msg.get("img_w", 1920), msg.get("img_h", 1080)
                )
        elif action == "set_target":
            if self._on_set_target:
                self._on_set_target(msg.get("x", 0), msg.get("y", 0), msg.get("theta", 0))
        elif action == "start_nav":
            if self._on_start_nav:
                self._on_start_nav()
        elif action == "stop_nav":
            if self._on_stop_nav:
                self._on_stop_nav()
        elif action == "clear_map":
            if self._on_clear_map:
                self._on_clear_map()
        elif action == "5":
            if self._on_cmd_5:
                self._on_cmd_5()
        elif action == "6":
            if self._on_cmd_6:
                self._on_cmd_6()

    # ---- 回调 ----
    _on_set_target = None
    _on_start_nav = None
    _on_stop_nav = None
    _on_clear_map = None
    _on_map_click = None
    _on_cmd_5 = None
    _on_cmd_6 = None

    def on_set_target(self, callback):
        self._on_set_target = callback

    def on_start_nav(self, callback):
        self._on_start_nav = callback

    def on_stop_nav(self, callback):
        self._on_stop_nav = callback

    def on_clear_map(self, callback):
        self._on_clear_map = callback

    def on_map_click(self, callback):
        self._on_map_click = callback

    def on_cmd_5(self, callback):
        self._on_cmd_5 = callback

    def on_cmd_6(self, callback):
        self._on_cmd_6 = callback

    # ==================== 启停 ====================

    def start(self):
        async def _serve():
            config = uvicorn.Config(
                self._app, host=self.host, port=self.port, log_level="warning"
            )
            server = uvicorn.Server(config)
            self._loop = asyncio.get_running_loop()
            await server.serve()

        def _run():
            asyncio.run(_serve())

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        print(f"[WS] 服务已启动: ws://{self.host}:{self.port}/ws"
              f" (最大客户端: {MAX_CLIENTS}, 图片宽度: {MAX_IMAGE_WIDTH}px)")

    def stop(self):
        pass
