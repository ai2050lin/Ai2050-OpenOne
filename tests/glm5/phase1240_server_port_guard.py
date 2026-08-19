"""Regression checks for the AI2050 backend port-start decision."""

from __future__ import annotations

import json
import socket
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.startup_guard import decide_server_start


class _HealthHandler(BaseHTTPRequestHandler):
    payload = {"status": "ok"}

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path != "/health":
            self.send_response(404)
            self.end_headers()
            return
        encoded = json.dumps(self.payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, _format, *_args):
        return


class ServerPortGuardTests(unittest.TestCase):
    def _serve(self, payload):
        handler = type("ConfiguredHealthHandler", (_HealthHandler,), {"payload": payload})
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server.server_address[1]

    def test_free_port_can_start(self):
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()

        decision = decide_server_start("127.0.0.1", port)

        self.assertTrue(decision.should_start)
        self.assertFalse(decision.existing_ai2050_backend)

    def test_existing_ai2050_backend_is_reused(self):
        port = self._serve(
            {
                "status": "ok",
                "model_loaded": False,
                "interceptor_ready": False,
            }
        )

        decision = decide_server_start("127.0.0.1", port)

        self.assertFalse(decision.should_start)
        self.assertTrue(decision.existing_ai2050_backend)
        self.assertIn("already running", decision.reason)

    def test_unrelated_service_is_rejected(self):
        port = self._serve({"status": "ok"})

        decision = decide_server_start("127.0.0.1", port)

        self.assertFalse(decision.should_start)
        self.assertFalse(decision.existing_ai2050_backend)
        self.assertIn("another service", decision.reason)

    def test_invalid_port_is_rejected(self):
        decision = decide_server_start("127.0.0.1", 70000)

        self.assertFalse(decision.should_start)
        self.assertIn("between 1 and 65535", decision.reason)


if __name__ == "__main__":
    unittest.main(verbosity=2)
