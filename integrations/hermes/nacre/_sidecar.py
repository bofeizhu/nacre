"""Client for the nacre sidecar (ndjson over a child process's stdio).

The sidecar (crates/nacre-node/sidecar/sidecar.mjs) owns the grit file and
the full nacre pipeline; this client just frames requests and matches
responses by id. Requests are answered strictly in order by the sidecar,
but the lock here keeps concurrent callers (worker thread + shutdown)
from interleaving writes.
"""

from __future__ import annotations

import json
import subprocess
import threading
from typing import Any, Dict, Optional


class SidecarError(RuntimeError):
    """A {id, error} response, or a dead/unresponsive sidecar."""


class SidecarClient:
    def __init__(self, node_bin: str, sidecar_path: str, env: Dict[str, str]):
        self._proc = subprocess.Popen(
            [node_bin, sidecar_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
            text=True,
            bufsize=1,  # line-buffered
        )
        self._lock = threading.Lock()
        self._next_id = 1

    def call(self, method: str, params: Optional[Dict[str, Any]] = None, *,
             timeout: float = 300.0) -> Dict[str, Any]:
        """One request, one response. Raises SidecarError on {error} or death."""
        with self._lock:
            if self._proc.poll() is not None:
                raise SidecarError(f"sidecar exited with code {self._proc.returncode}")
            req_id = self._next_id
            self._next_id += 1
            line = json.dumps({"id": req_id, "method": method, "params": params or {}})
            try:
                self._proc.stdin.write(line + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                raise SidecarError(f"sidecar pipe broken: {e}") from e
            # The sidecar answers in order; the next line is our response.
            timer = threading.Timer(timeout, self._proc.kill)
            timer.start()
            try:
                raw = self._proc.stdout.readline()
            finally:
                timer.cancel()
            if not raw:
                raise SidecarError("sidecar closed stdout (killed or crashed)")
            msg = json.loads(raw)
            if msg.get("id") != req_id:
                raise SidecarError(f"response id mismatch: sent {req_id}, got {msg.get('id')}")
            if "error" in msg:
                raise SidecarError(msg["error"])
            return msg.get("result") or {}

    def close(self) -> None:
        """Best-effort clean shutdown; escalates to kill."""
        try:
            self.call("shutdown", {}, timeout=10.0)
        except Exception:
            pass
        try:
            self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.wait(timeout=5.0)
        except Exception:
            self._proc.kill()

    @property
    def alive(self) -> bool:
        return self._proc.poll() is None
