"""
WebSocket 服务：将 PyQt5 GUI 界面截图实时推送到前端
使用 FastAPI + asyncio，在独立线程中运行，不阻塞 PyQt5 主循环
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
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QBuffer, QIODevice, QRect

MAX_CLIENTS = 20
MAX_IMAGE_WIDTH = 1280


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
        self._client_counter = 0

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

            while len(self._clients) >= MAX_CLIENTS:
                old = next(iter(self._clients))
                self._clients.discard(old)
                asyncio.create_task(self._safe_close(old))

            self._client_counter += 1
            cid = self._client_counter
            self._clients.add(ws)
            print(f"[WS] #{cid} 已连接 (共 {len(self._clients)} 个)")

            try:
                while True:
                    await ws.receive_text()  # 只接收不处理，保持连接
            except WebSocketDisconnect:
                pass
            except Exception:
                pass
            finally:
                self._clients.discard(ws)
                print(f"[WS] #{cid} 已断开 (共 {len(self._clients)} 个)")

    def set_main_window(self, window: QMainWindow):
        self._main_window = window

    # ==================== 广播方法 ====================

    def broadcast_window_image(self, quality: int = 95, crop_rect: Optional[QRect] = None):
        """截取主窗口 → 裁剪(可选) → 缩放 → JPEG → base64 → JSON 广播"""
        if not self._main_window or not self._clients:
            return

        pixmap = self._main_window.grab()
        qimage = pixmap.toImage().convertToFormat(QImage.Format_RGB888)

        if crop_rect is not None and crop_rect.isValid():
            x = max(0, crop_rect.x())
            y = max(0, crop_rect.y())
            w = min(crop_rect.width(), qimage.width() - x)
            h = min(crop_rect.height(), qimage.height() - y)
            qimage = qimage.copy(x, y, w, h)

        if qimage.width() > MAX_IMAGE_WIDTH:
            ratio = MAX_IMAGE_WIDTH / qimage.width()
            new_h = int(qimage.height() * ratio)
            qimage = qimage.scaled(
                MAX_IMAGE_WIDTH, new_h,
                aspectRatioMode=1,
                transformMode=1
            )

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
        try:
            await ws.close()
        except Exception:
            pass

    async def _broadcast_text(self, text: str):
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
        print(f"[WS] 服务已启动: ws://{self.host}:{self.port}/ws")

    def stop(self):
        pass
