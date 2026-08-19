"""Small, dependency-free checks used before starting the API server."""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


@dataclass(frozen=True)
class PortDecision:
    should_start: bool
    reason: str
    existing_ai2050_backend: bool = False


def _health_host(bind_host: str) -> str:
    if bind_host in {"", "0.0.0.0", "::", "[::]"}:
        return "127.0.0.1"
    return bind_host


def _port_is_available(bind_host: str, port: int) -> bool:
    """Attempt the same IPv4 bind Uvicorn will use, then release it."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        probe.bind((bind_host, port))
        return True
    except OSError:
        return False
    finally:
        probe.close()


def _is_ai2050_backend(bind_host: str, port: int, timeout: float) -> bool:
    url = f"http://{_health_host(bind_host)}:{port}/health"
    try:
        with urlopen(url, timeout=timeout) as response:
            if response.status != 200:
                return False
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
        return False

    return (
        isinstance(payload, dict)
        and payload.get("status") == "ok"
        and "model_loaded" in payload
        and "interceptor_ready" in payload
    )


def decide_server_start(bind_host: str, port: int, timeout: float = 0.75) -> PortDecision:
    """Return a user-facing decision without starting or stopping any process."""
    if not 1 <= port <= 65535:
        return PortDecision(False, f"AI2050_PORT must be between 1 and 65535; got {port}.")

    if _port_is_available(bind_host, port):
        return PortDecision(True, f"Port {bind_host}:{port} is available.")

    endpoint = f"http://{_health_host(bind_host)}:{port}"
    if _is_ai2050_backend(bind_host, port, timeout):
        return PortDecision(
            False,
            f"AI2050 backend is already running at {endpoint}; reusing the existing server.",
            existing_ai2050_backend=True,
        )

    return PortDecision(
        False,
        (
            f"Cannot start AI2050: {bind_host}:{port} is already used by another service. "
            "Stop that process, or set AI2050_PORT to a free port and update the frontend "
            "VITE_API_BASE/VITE_ANALYSIS_API_BASE values to the same port."
        ),
    )
