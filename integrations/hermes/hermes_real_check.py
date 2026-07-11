"""Real-Hermes integration check — run UNDER HERMES'S OWN venv python.

Exercises the seams the offline pytest (which stubs the ABC) cannot:
Hermes's plugin discovery, its loader, its real MemoryProvider ABC, and
its MemoryManager orchestration (initialize_all with cli extras,
sync_all's background worker + signature inspection, teardown) — all in
replay mode against the committed trace1 recordings. Offline; no keys.

Invoked by tests/test_hermes_real.py via subprocess (skipped when no
Hermes install is present). Exit code 0 = pass.

Usage: $HERMES_HOME/hermes-agent/venv/bin/python hermes_real_check.py <hermes-agent-dir>
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

HERMES_AGENT = sys.argv[1] if len(sys.argv) > 1 else str(Path.home() / ".hermes" / "hermes-agent")
sys.path.insert(0, HERMES_AGENT)

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "oracle" / "fixtures" / "trace1"
NODE24 = Path.home() / ".nvm" / "versions" / "node" / "v24.18.0" / "bin"

os.environ.update(
    {
        "NACRE_LLM_PROVIDER": "replay",
        "NACRE_LLM_RECORDINGS": str(FIXTURES / "llm_recordings.json"),
        "NACRE_EMBEDDER_PROVIDER": "replay",
        "NACRE_EMBEDDER_RECORDINGS": str(FIXTURES / "embedder_recordings.json"),
    }
)
if NODE24.is_dir():
    os.environ["PATH"] = f"{NODE24}{os.pathsep}" + os.environ["PATH"]


def main() -> None:
    spec = json.loads((REPO / "oracle" / "episodes" / "trace1.json").read_text())
    ep0 = spec["episodes"][0]["content"]
    user_label, rest = ep0.split(": ", 1)
    user_content, assistant_part = rest.split("\nPriya: ", 1)

    # --- A: discovery + loader + real ABC -----------------------------------
    from plugins.memory import discover_memory_providers, load_memory_provider

    found = {name: avail for name, _, avail in discover_memory_providers()}
    assert found.get("nacre"), f"nacre not discovered/available: {found}"
    print("A1. discovered by Hermes's loader (available=True)")

    from agent.memory_provider import MemoryProvider

    probe = load_memory_provider("nacre")
    assert isinstance(probe, MemoryProvider), type(probe).__mro__
    assert probe.system_prompt_block() == "" and probe.get_tool_schemas() == []
    print("A2. loader instantiates a real-ABC subclass; Stage-1 surface invisible")

    # --- B: MemoryManager orchestration (the production seam) ---------------
    from agent.memory_manager import MemoryManager

    home = Path(tempfile.mkdtemp(prefix="hermes-real-check-"))
    (home / "nacre.json").write_text(
        json.dumps(
            {"group_id": spec["group_id"], "user_label": user_label, "assistant_label": "Priya"}
        )
    )
    mm = MemoryManager()
    p = load_memory_provider("nacre")
    p._clock = lambda: datetime.fromisoformat(spec["episodes"][0]["reference_time"])
    mm.add_provider(p)
    mm.initialize_all(
        session_id="hermes-real-check",
        platform="cli",
        hermes_home=str(home),
        agent_context="primary",
        warning_callback=lambda *a, **k: None,
        status_callback=lambda *a, **k: None,
        session_title="real check",
        agent_identity="default",
        agent_workspace="hermes",
    )
    assert p._active, "initialize_all did not activate the provider"
    print("B1. MemoryManager.initialize_all (with cli extras) -> active")

    mm.sync_all(
        user_content,
        assistant_part,
        session_id="hermes-real-check",
        messages=[
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_part},
        ],
    )
    deadline = time.time() + 60
    while time.time() < deadline and p.last_result is None:
        time.sleep(0.2)
    r = p.last_result
    assert r and len(r["nodeIds"]) == 4 and len(r["newEdgeIds"]) == 2, f"bad deltas: {r}"
    assert p.stats()["episodes"] == 1
    print("B2. sync_all background worker -> ep-0 deltas exact, episode landed")

    mm.on_session_end([])
    mm.shutdown_all()
    assert not p._active
    print("B3. on_session_end + shutdown_all -> clean teardown")
    print("PASS")


if __name__ == "__main__":
    main()
