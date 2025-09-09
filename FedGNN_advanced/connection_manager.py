# connection_manager.py
from fastapi import WebSocket
from typing import List
import asyncio
import logging

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manage active WebSocket connections and broadcast messages."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
        logger.info("WebSocket connected, total=%d", len(self.active_connections))

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        logger.info("WebSocket disconnected, total=%d", len(self.active_connections))

    async def send_message(self, message: str):
        """Broadcast a message to all active websockets. Returns when sends scheduled."""
        async with self._lock:
            conns = list(self.active_connections)
        if not conns:
            return
        # send concurrently, swallow per-connection exceptions
        coros = [self._safe_send(ws, message) for ws in conns]
        await asyncio.gather(*coros, return_exceptions=True)

    async def _safe_send(self, websocket: WebSocket, message: str):
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.warning("WebSocket send failed, removing conn: %s", e)
            await self.disconnect(websocket)
