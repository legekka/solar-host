"""WebSocket client for connecting to solar-control.

This module provides a persistent WebSocket connection to solar-control,
handling:
- Registration on connect
- Reconnection with exponential backoff
- Event streaming (logs, instance state, health)
- Event queuing during disconnection
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from collections import deque
from enum import Enum

try:
    import websockets
    from websockets.client import WebSocketClientProtocol
except ImportError:
    websockets = None  # type: ignore
    WebSocketClientProtocol = None  # type: ignore


class WSMessageType(str, Enum):
    """WebSocket message types matching solar-control protocol."""

    REGISTRATION = "registration"
    LOG = "log"
    INSTANCE_STATE = "instance_state"
    HOST_HEALTH = "host_health"


class SolarControlClient:
    """WebSocket client for maintaining persistent connection to solar-control."""

    def __init__(
        self,
        control_url: str,
        host_id: str,
        api_key: str,
        host_name: str = "",
        reconnect_delay: float = 1.0,
        max_reconnect_delay: float = 30.0,
        ping_interval: float = 25.0,
        max_queue_size: int = 1000,
    ):
        self.control_url = control_url
        self.host_id = host_id
        self.api_key = api_key
        self.host_name = host_name or host_id
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay
        self.ping_interval = ping_interval
        self.max_queue_size = max_queue_size

        self._ws: Optional[WebSocketClientProtocol] = None
        self._connected = False
        self._running = False
        self._current_delay = reconnect_delay
        self._event_queue: deque = deque(maxlen=max_queue_size)
        self._lock = asyncio.Lock()
        self._connection_task: Optional[asyncio.Task] = None
        self._ping_task: Optional[asyncio.Task] = None
        self._queue_task: Optional[asyncio.Task] = None

    @property
    def is_connected(self) -> bool:
        """Check if currently connected to solar-control."""
        return self._connected and self._ws is not None

    async def start(self):
        """Start the WebSocket client and connect to solar-control."""
        if not self.control_url:
            print("SolarControlClient: No control URL configured, skipping connection")
            return

        if websockets is None:
            print(
                "SolarControlClient: websockets library not installed, skipping connection"
            )
            return

        self._running = True
        self._connection_task = asyncio.create_task(self._connection_loop())
        print(f"SolarControlClient: Starting connection to {self.control_url}")

    async def stop(self):
        """Stop the WebSocket client and disconnect."""
        self._running = False

        if self._ping_task:
            self._ping_task.cancel()
            try:
                await self._ping_task
            except asyncio.CancelledError:
                pass

        if self._queue_task:
            self._queue_task.cancel()
            try:
                await self._queue_task
            except asyncio.CancelledError:
                pass

        if self._connection_task:
            self._connection_task.cancel()
            try:
                await self._connection_task
            except asyncio.CancelledError:
                pass

        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass

        self._connected = False
        print("SolarControlClient: Stopped")

    async def _connection_loop(self):
        """Main connection loop with reconnection logic."""
        while self._running:
            try:
                await self._connect_and_run()
            except Exception as e:
                if self._running:
                    print(f"SolarControlClient: Connection error: {e}")

            if self._running:
                # Exponential backoff for reconnection
                print(
                    f"SolarControlClient: Reconnecting in {self._current_delay:.1f}s..."
                )
                await asyncio.sleep(self._current_delay)
                self._current_delay = min(
                    self._current_delay * 2, self.max_reconnect_delay
                )

    async def _connect_and_run(self):
        """Connect to solar-control and run the message loop."""
        if websockets is None:
            return

        async with websockets.connect(self.control_url) as ws:
            self._ws = ws
            print(f"SolarControlClient: Connected to {self.control_url}")

            # Send registration
            await self._send_registration()

            # Wait for registration acknowledgement
            try:
                response = await asyncio.wait_for(ws.recv(), timeout=10.0)
                msg = json.loads(response)
                if msg.get("type") == "error":
                    raise Exception(f"Registration failed: {msg.get('message')}")
                if msg.get("type") != "registration_ack":
                    raise Exception(f"Unexpected response: {msg}")
                print(f"SolarControlClient: Registered as host '{self.host_id}'")
            except asyncio.TimeoutError:
                raise Exception("Registration acknowledgement timeout")

            self._connected = True
            self._current_delay = self.reconnect_delay  # Reset backoff on success

            # Start ping and queue drain tasks
            self._ping_task = asyncio.create_task(self._ping_loop())
            self._queue_task = asyncio.create_task(self._drain_queue())

            # Run receive loop
            try:
                async for message in ws:
                    if message == "pong":
                        continue
                    # Handle any incoming messages from solar-control
                    # (currently just keepalives)
                    try:
                        msg = json.loads(message)
                        if msg.get("type") == "keepalive":
                            continue
                    except json.JSONDecodeError:
                        pass

            except Exception as e:
                print(f"SolarControlClient: Receive loop error: {e}")
            finally:
                self._connected = False
                if self._ping_task:
                    self._ping_task.cancel()
                if self._queue_task:
                    self._queue_task.cancel()

    async def _send_registration(self):
        """Send registration message to solar-control."""
        if not self._ws:
            return

        # Get current instances info
        from app.config import config_manager

        instances = []
        for instance in config_manager.get_all_instances():
            instances.append(
                {
                    "id": instance.id,
                    "alias": instance.config.alias,
                    "status": instance.status.value,
                    "port": instance.port,
                }
            )

        registration = {
            "type": WSMessageType.REGISTRATION.value,
            "data": {
                "host_id": self.host_id,
                "api_key": self.api_key,
                "host_name": self.host_name,
                "instances": instances,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        await self._ws.send(json.dumps(registration))

    async def _ping_loop(self):
        """Send periodic pings to keep connection alive."""
        while self._connected and self._ws:
            try:
                await asyncio.sleep(self.ping_interval)
                if self._ws and self._connected:
                    await self._ws.send("ping")
            except asyncio.CancelledError:
                break
            except Exception:
                break

    async def _drain_queue(self):
        """Drain queued events when connected."""
        while self._connected:
            try:
                async with self._lock:
                    while self._event_queue and self._ws and self._connected:
                        event = self._event_queue.popleft()
                        try:
                            await self._ws.send(json.dumps(event))
                        except Exception:
                            # Put it back and break
                            self._event_queue.appendleft(event)
                            break
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(0.5)

    async def send_event(self, event: Dict[str, Any]):
        """Send an event to solar-control, queuing if disconnected."""
        async with self._lock:
            if self._connected and self._ws:
                try:
                    await self._ws.send(json.dumps(event))
                    return
                except Exception:
                    pass

            # Queue the event for later
            self._event_queue.append(event)

    async def send_log(
        self, instance_id: str, seq: int, line: str, level: str = "info"
    ):
        """Send a log message to solar-control."""
        event = {
            "type": WSMessageType.LOG.value,
            "instance_id": instance_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "seq": seq,
                "line": line,
                "level": level,
            },
        }
        await self.send_event(event)

    async def send_instance_state(self, instance_id: str, state: Dict[str, Any]):
        """Send instance runtime state update to solar-control."""
        event = {
            "type": WSMessageType.INSTANCE_STATE.value,
            "instance_id": instance_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": state,
        }
        await self.send_event(event)

    async def send_health(self, memory: Optional[Dict[str, Any]] = None):
        """Send host health/memory update to solar-control."""
        from app.memory_monitor import get_memory_info
        from app.config import config_manager

        if memory is None:
            memory = get_memory_info()

        instances = config_manager.get_all_instances()
        running_count = sum(1 for i in instances if i.status.value == "running")

        event = {
            "type": WSMessageType.HOST_HEALTH.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "memory": memory,
                "instance_count": len(instances),
                "running_instance_count": running_count,
            },
        }
        await self.send_event(event)


# Global client instance (initialized in main.py)
solar_control_client: Optional[SolarControlClient] = None


def get_client() -> Optional[SolarControlClient]:
    """Get the global solar-control client."""
    return solar_control_client


def init_client(settings) -> Optional[SolarControlClient]:
    """Initialize the global solar-control client from settings."""
    global solar_control_client

    if not settings.solar_control_url:
        print("SolarControlClient: SOLAR_CONTROL_URL not configured")
        return None

    if not settings.host_id:
        print("SolarControlClient: HOST_ID not configured")
        return None

    solar_control_client = SolarControlClient(
        control_url=settings.solar_control_url,
        host_id=settings.host_id,
        api_key=settings.solar_control_api_key,
        host_name=settings.host_name or settings.host_id,
        reconnect_delay=settings.ws_reconnect_delay,
        max_reconnect_delay=settings.ws_reconnect_max_delay,
        ping_interval=settings.ws_ping_interval,
    )

    return solar_control_client
