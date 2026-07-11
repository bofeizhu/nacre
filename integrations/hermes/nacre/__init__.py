"""nacre memory provider for Hermes — Stage 1, CAPTURE-ONLY.

Every completed turn is ingested into a local bi-temporal memory graph
(nacre pipeline over a grit/SQLite file) via a supervised Node sidecar.
Deliberately writes-only: no prefetch, no tools, empty system prompt —
zero influence on any Hermes session. Recall is Stage 2, gated on an
offline review of real captured graphs.

The provider owns its LLM/embedder calls (DeepSeek + Zhipu by default) on
its OWN keys — separate from Hermes's chat model. ~5-10 small-model calls
per turn, queued on a worker thread; a circuit breaker fails OPEN (memory
stops, the session never breaks).

Layout: this package is symlinked into $HERMES_HOME/plugins/nacre by
integrations/hermes/install.sh; the sidecar + addon live in the nacre
repo the symlink points into.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

from ._sidecar import SidecarClient, SidecarError

logger = logging.getLogger(__name__)

_BREAKER_STRIKES = 3
_QUEUE_MAX = 256  # bounded: drop-oldest beats unbounded growth on outage


def _default_nacre_node_dir() -> Path:
    """The nacre-node crate dir, resolved through the install symlink.

    realpath(__file__) lands in the repo checkout
    (<repo>/integrations/hermes/nacre/__init__.py), so the crate is three
    levels up. A copied (non-symlinked) install must set nacre_node_dir in
    config instead.
    """
    return Path(__file__).resolve().parents[3] / "crates" / "nacre-node"


def _load_config(hermes_home: Optional[str] = None) -> Dict[str, Any]:
    """Read $HERMES_HOME/nacre.json (written by save_config); {} if absent."""
    home = hermes_home or os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    path = Path(home) / "nacre.json"
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


class NacreMemoryProvider(MemoryProvider):
    """Capture-only provider: sync_turn → nacre addEpisode. Nothing flows back."""

    def __init__(self, clock=None):
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._client: Optional[SidecarClient] = None
        self._queue: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue(maxsize=_QUEUE_MAX)
        self._worker: Optional[threading.Thread] = None
        self._active = False  # initialized, primary context, breaker closed
        self._config: Dict[str, Any] = {}
        self._group_id = "hermes"
        # Circuit breaker: fail OPEN — stop ingesting, never break the session.
        self._consecutive_failures = 0
        self._breaker_open = False
        self._breaker_lock = threading.Lock()
        # Last addEpisode result, for tests and debug logging only.
        self.last_result: Optional[Dict[str, Any]] = None

    # -- identity / availability --------------------------------------------

    @property
    def name(self) -> str:
        return "nacre"

    def is_available(self) -> bool:
        """Config + files only — no network, no process spawn (ABC contract)."""
        cfg = _load_config()
        has_llm_key = bool(os.environ.get("NACRE_LLM_API_KEY") or cfg.get("llm_api_key"))
        has_emb_key = bool(os.environ.get("NACRE_EMBEDDER_API_KEY") or cfg.get("embedder_api_key"))
        replay = os.environ.get("NACRE_LLM_PROVIDER") == "replay"
        node_dir = Path(cfg.get("nacre_node_dir") or _default_nacre_node_dir())
        sidecar_ok = (node_dir / "sidecar" / "sidecar.mjs").exists() and (
            node_dir / "index.js"
        ).exists()
        node_ok = shutil.which(cfg.get("node_bin", "node")) is not None
        return ((has_llm_key and has_emb_key) or replay) and sidecar_ok and node_ok

    # -- setup contract ------------------------------------------------------

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "llm_api_key",
                "description": "DeepSeek API key for nacre's extraction pipeline (its own budget, separate from the chat model)",
                "secret": True,
                "required": True,
                "env_var": "NACRE_LLM_API_KEY",
                "url": "https://platform.deepseek.com/api_keys",
            },
            {
                "key": "embedder_api_key",
                "description": "Zhipu API key for embedding-3 (nacre's retrieval embeddings)",
                "secret": True,
                "required": True,
                "env_var": "NACRE_EMBEDDER_API_KEY",
                "url": "https://open.bigmodel.cn",
            },
            {
                "key": "group_id",
                "description": "Memory graph namespace",
                "default": "hermes",
            },
            {
                "key": "node_bin",
                "description": "Node.js >= 20 binary for the sidecar",
                "default": "node",
            },
            {
                "key": "nacre_node_dir",
                "description": "Path to the built nacre-node crate (blank = resolve through the install symlink)",
                "required": False,
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        path = Path(hermes_home) / "nacre.json"
        existing: Dict[str, Any] = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text())
            except Exception:
                pass
        existing.update({k: v for k, v in values.items() if v not in (None, "")})
        try:
            from utils import atomic_json_write  # inside Hermes

            atomic_json_write(path, existing, mode=0o600)
        except Exception:
            path.write_text(json.dumps(existing, indent=1) + "\n")
            os.chmod(path, 0o600)

    # -- lifecycle -----------------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        context = kwargs.get("agent_context", "primary")
        if context != "primary":
            logger.debug("nacre: skipping non-primary context %r", context)
            return
        hermes_home = kwargs.get("hermes_home") or os.environ.get("HERMES_HOME") or str(
            Path.home() / ".hermes"
        )
        self._config = _load_config(hermes_home)
        self._group_id = self._config.get("group_id", "hermes")
        node_dir = Path(self._config.get("nacre_node_dir") or _default_nacre_node_dir())
        db_dir = Path(hermes_home) / "nacre"
        db_dir.mkdir(parents=True, exist_ok=True)

        env = dict(os.environ)
        env.setdefault("NACRE_LLM_PROVIDER", "deepseek")
        env.setdefault("NACRE_EMBEDDER_PROVIDER", "zhipu")
        for env_name, cfg_key in (
            ("NACRE_LLM_API_KEY", "llm_api_key"),
            ("NACRE_EMBEDDER_API_KEY", "embedder_api_key"),
        ):
            if not env.get(env_name) and self._config.get(cfg_key):
                env[env_name] = str(self._config[cfg_key])

        try:
            self._client = SidecarClient(
                self._config.get("node_bin", "node"),
                str(node_dir / "sidecar" / "sidecar.mjs"),
                env,
            )
            self._client.call(
                "init",
                {
                    "dbPath": str(db_dir / "memory.db"),
                    "deviceId": f"hermes-{kwargs.get('agent_identity', 'default')}",
                    "groupId": self._group_id,
                },
                timeout=30.0,
            )
        except Exception as e:
            logger.warning("nacre: sidecar failed to start — memory capture disabled: %s", e)
            self._trip_breaker()
            return

        self._worker = threading.Thread(target=self._drain, name="nacre-sync", daemon=True)
        self._worker.start()
        self._active = True
        logger.info("nacre: capture-only memory active (group %r)", self._group_id)

    def system_prompt_block(self) -> str:
        return ""  # Stage 1: invisible by design

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return []  # Stage 1: no tools

    # -- capture -------------------------------------------------------------

    def sync_turn(self, user_content: str, assistant_content: str, *,
                  session_id: str = "", messages=None) -> None:
        if not self._active or self._breaker_open:
            return
        if not (user_content or "").strip() and not (assistant_content or "").strip():
            return
        episode = {
            "content": self._format_episode(user_content or "", assistant_content or ""),
            "source": "message",
            "sourceDescription": self._config.get("source_description", "hermes chat"),
            "validAt": self._clock().isoformat(),
        }
        try:
            self._queue.put_nowait(episode)
        except queue.Full:
            logger.warning("nacre: ingestion queue full — dropping oldest turn")
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                pass
            self._queue.put_nowait(episode)

    def _format_episode(self, user_content: str, assistant_content: str) -> str:
        """Graphiti message convention: 'speaker: text' lines."""
        user_label = self._config.get("user_label", "user")
        assistant_label = self._config.get("assistant_label", "assistant")
        return f"{user_label}: {user_content}\n{assistant_label}: {assistant_content}"

    def _drain(self) -> None:
        while True:
            episode = self._queue.get()
            if episode is None:  # shutdown sentinel
                self._queue.task_done()
                return
            try:
                self.last_result = self._client.call("addEpisode", episode)
                self._record_success()
            except Exception as e:
                self._record_failure(e)
            finally:
                self._queue.task_done()

    # -- circuit breaker (fail OPEN: stop ingesting, never break the session) --

    def _record_success(self) -> None:
        with self._breaker_lock:
            self._consecutive_failures = 0

    def _record_failure(self, e: Exception) -> None:
        with self._breaker_lock:
            self._consecutive_failures += 1
            logger.warning(
                "nacre: addEpisode failed (%d/%d): %s",
                self._consecutive_failures, _BREAKER_STRIKES, e,
            )
            if self._consecutive_failures >= _BREAKER_STRIKES:
                self._trip_breaker()

    def _trip_breaker(self) -> None:
        if not self._breaker_open:
            self._breaker_open = True
            self._active = False
            logger.error(
                "nacre: circuit breaker OPEN after %d consecutive failures — "
                "memory capture disabled for this session (chat unaffected)",
                self._consecutive_failures,
            )

    # -- helpers / teardown ----------------------------------------------------

    def flush(self, timeout: float = 600.0) -> None:
        """Block until queued episodes are ingested (tests and shutdown)."""
        done = threading.Event()
        threading.Thread(target=lambda: (self._queue.join(), done.set()), daemon=True).start()
        done.wait(timeout)

    def stats(self) -> Dict[str, Any]:
        """Sidecar status (episode/node/edge counts) — debug surface."""
        if not self._client or not self._client.alive:
            return {"initialized": False}
        return self._client.call("status", {}, timeout=10.0)

    def backup_paths(self) -> List[str]:
        return []  # the db lives under HERMES_HOME; `hermes backup` walks it

    def shutdown(self) -> None:
        if self._worker and self._worker.is_alive():
            self.flush(timeout=30.0)
            self._queue.put(None)
            self._worker.join(timeout=5.0)
        if self._client:
            self._client.close()
            self._client = None
        self._active = False
