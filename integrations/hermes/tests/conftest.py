"""Test scaffolding: run the plugin OUTSIDE Hermes.

The plugin imports ``agent.memory_provider.MemoryProvider`` (a Hermes
module). Tests stub that module with a minimal structural copy of the ABC
(same method names, no behavior) so the plugin imports cleanly, then load
the plugin package from ../nacre the same way Hermes's loader does
(spec_from_file_location + registered submodules).

The sidecar needs Node >= 20; a stale nvm default (v16) shadows v24 on
this machine, so a known-good bin dir is prepended when the PATH node is
too old.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import types
from abc import ABC, abstractmethod
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PLUGIN_DIR = HERE.parent / "nacre"
REPO_ROOT = HERE.parents[2]
NACRE_NODE = REPO_ROOT / "crates" / "nacre-node"
FIXTURES = REPO_ROOT / "oracle" / "fixtures" / "trace1"

_KNOWN_NODE24 = Path.home() / ".nvm" / "versions" / "node" / "v24.18.0" / "bin"


def _ensure_modern_node() -> bool:
    """True if a Node >= 20 is reachable; fixes PATH if a known v24 exists."""
    node = shutil.which("node")
    if node:
        try:
            major = int(
                subprocess.run([node, "--version"], capture_output=True, text=True)
                .stdout.strip()
                .lstrip("v")
                .split(".")[0]
            )
            if major >= 20:
                return True
        except Exception:
            pass
    if _KNOWN_NODE24.is_dir():
        os.environ["PATH"] = f"{_KNOWN_NODE24}{os.pathsep}" + os.environ.get("PATH", "")
        return True
    return False


def _stub_hermes_abc() -> None:
    """Install a structural stand-in for agent.memory_provider."""
    if "agent.memory_provider" in sys.modules:
        return

    class MemoryProvider(ABC):  # mirrors the Hermes ABC's surface
        @property
        @abstractmethod
        def name(self) -> str: ...

        @abstractmethod
        def is_available(self) -> bool: ...

        @abstractmethod
        def initialize(self, session_id: str, **kwargs) -> None: ...

        @abstractmethod
        def get_tool_schemas(self): ...

        def system_prompt_block(self) -> str:
            return ""

        def prefetch(self, query: str, *, session_id: str = "") -> str:
            return ""

        def queue_prefetch(self, query: str, *, session_id: str = "") -> None: ...

        def sync_turn(self, user_content, assistant_content, *, session_id="", messages=None): ...

        def handle_tool_call(self, tool_name, args, **kwargs) -> str:
            raise NotImplementedError

        def shutdown(self) -> None: ...

        def on_turn_start(self, turn_number, message, **kwargs) -> None: ...

        def on_session_end(self, messages) -> None: ...

        def on_session_switch(self, new_session_id, **kwargs) -> None: ...

        def on_pre_compress(self, messages) -> str:
            return ""

        def on_delegation(self, task, result, **kwargs) -> None: ...

        def on_memory_write(self, action, target, content, metadata=None) -> None: ...

        def get_config_schema(self):
            return []

        def save_config(self, values, hermes_home) -> None: ...

        def backup_paths(self):
            return []

    agent_pkg = types.ModuleType("agent")
    mp_mod = types.ModuleType("agent.memory_provider")
    mp_mod.MemoryProvider = MemoryProvider
    agent_pkg.memory_provider = mp_mod
    sys.modules["agent"] = agent_pkg
    sys.modules["agent.memory_provider"] = mp_mod


def _load_plugin():
    """Load the plugin package the way Hermes's loader does."""
    _stub_hermes_abc()
    name = "_test_nacre_plugin"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, PLUGIN_DIR / "__init__.py", submodule_search_locations=[str(PLUGIN_DIR)]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def plugin():
    if not (NACRE_NODE / "index.js").exists():
        pytest.skip("nacre-node addon not built — run `npm run build` first")
    if not _ensure_modern_node():
        pytest.skip("no Node >= 20 available for the sidecar")
    return _load_plugin()


@pytest.fixture()
def replay_env(monkeypatch):
    """Point the sidecar at the committed trace1 recordings (offline)."""
    monkeypatch.setenv("NACRE_LLM_PROVIDER", "replay")
    monkeypatch.setenv("NACRE_LLM_RECORDINGS", str(FIXTURES / "llm_recordings.json"))
    monkeypatch.setenv("NACRE_EMBEDDER_PROVIDER", "replay")
    monkeypatch.setenv("NACRE_EMBEDDER_RECORDINGS", str(FIXTURES / "embedder_recordings.json"))
