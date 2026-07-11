"""Run the real-Hermes integration check when a Hermes install exists.

The check must execute under HERMES'S venv python (its dependency tree,
its Python version), so this test shells out rather than importing.
Skipped cleanly on machines without Hermes — the stub-based tests in
test_provider.py still cover the provider itself everywhere.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
CHECK = HERE.parent / "hermes_real_check.py"
HERMES_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
HERMES_AGENT = HERMES_HOME / "hermes-agent"


def _hermes_python() -> Path | None:
    for name in ("venv", ".venv"):
        candidate = HERMES_AGENT / name / "bin" / "python"
        if candidate.exists():
            return candidate
    return None


def test_real_hermes_loader_and_memory_manager(plugin):  # plugin: addon+node guard
    python = _hermes_python()
    if python is None:
        pytest.skip("no Hermes install on this machine")
    if not (HERMES_HOME / "plugins" / "nacre").exists():
        pytest.skip("nacre plugin not installed into Hermes — run install.sh")
    result = subprocess.run(
        [str(python), str(CHECK), str(HERMES_AGENT)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "PASS" in result.stdout
