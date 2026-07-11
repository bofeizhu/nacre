"""Live smoke of the Hermes provider code path — OUTSIDE Hermes, real APIs.

Never run by pytest (no test_ prefix). Spawns the real sidecar with live
DeepSeek + Zhipu credentials mapped from oracle/.env (values are set in
the process environment only — never printed), drives 3 synthetic
conversation turns through NacreMemoryProvider.sync_turn exactly as
Hermes would, then reports counts and retrieval results.

Usage:  uv run python live_smoke.py [hermes_home_dir]
        (default hermes_home: a fresh temp dir; pass a path to keep the db)
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "tests"))
from conftest import _ensure_modern_node, _load_plugin  # noqa: E402

REPO_ROOT = HERE.parents[1]
ORACLE_ENV = REPO_ROOT / "oracle" / ".env"

TURNS = [
    (
        "I just adopted a corgi named Waffle from the shelter in Astoria.",
        "Congratulations! Corgis are wonderful — how is Waffle settling in?",
    ),
    (
        "Great. Waffle's vet is Dr. Ilse Braun at the Greenpoint Animal Clinic.",
        "Good to know — Dr. Ilse Braun at Greenpoint Animal Clinic, noted.",
    ),
    (
        "Update: we switched vets last week. Waffle now sees Dr. Marco Reyes, "
        "still at the Greenpoint Animal Clinic.",
        "Got it — Waffle's vet is now Dr. Marco Reyes at Greenpoint Animal Clinic.",
    ),
]

QUERIES = ["Who is Waffle's vet?", "Where was Waffle adopted?"]


def load_live_env() -> None:
    """Map oracle/.env capture keys to the provider's env vars. No echoes."""
    import os

    if not ORACLE_ENV.exists():
        sys.exit("oracle/.env not found — live smoke needs the capture keys")
    values = {}
    for line in ORACLE_ENV.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            values[k.strip()] = v.strip()
    os.environ["NACRE_LLM_PROVIDER"] = "deepseek"
    os.environ["NACRE_LLM_API_KEY"] = values["CAPTURE_LLM_API_KEY"]
    os.environ["NACRE_EMBEDDER_PROVIDER"] = "zhipu"
    os.environ["NACRE_EMBEDDER_API_KEY"] = values["CAPTURE_EMBEDDER_API_KEY"]


def main() -> None:
    if not _ensure_modern_node():
        sys.exit("no Node >= 20 available")
    load_live_env()
    plugin = _load_plugin()

    home = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="nacre-smoke-"))
    home.mkdir(parents=True, exist_ok=True)
    (home / "nacre.json").write_text(json.dumps({"group_id": "hermes-smoke"}))

    provider = plugin.NacreMemoryProvider()
    provider.initialize(
        "live-smoke-session",
        hermes_home=str(home),
        platform="cli",
        agent_context="primary",
        agent_identity="smoke",
    )
    if not provider._active:
        sys.exit("provider failed to initialize — see warnings above")

    t0 = time.time()
    for i, (user, assistant) in enumerate(TURNS):
        provider.sync_turn(user, assistant)
        provider.flush()  # serialize so per-turn results are attributable
        r = provider.last_result
        if r is None:
            sys.exit(f"turn {i}: ingestion failed (breaker: {provider._breaker_open})")
        print(
            f"turn {i}: +{len(r['nodeIds'])} nodes, +{len(r['newEdgeIds'])} edges, "
            f"{len(r['merges'])} merges, {len(r['invalidatedEdgeIds'])} invalidated"
        )

    stats = provider.stats()
    print(
        f"\ngraph after {len(TURNS)} turns ({time.time() - t0:.0f}s): "
        f"{stats['episodes']} episodes, {stats['liveNodes']} live nodes, {stats['edges']} edges"
    )

    for q in QUERIES:
        hits = provider._client.call("searchEdges", {"query": q, "limit": 3})["hits"]
        print(f"\nQ: {q}")
        for h in hits:
            # napi omits null optionals from the JSON entirely — use .get().
            invalid_at = h.get("invalidAt")
            current = "current" if not invalid_at else f"invalid since {invalid_at[:10]}"
            print(f"   - {h['fact']}  [{current}; episodes: {len(h['episodes'])}]")

    provider.shutdown()
    print(f"\ndb kept at: {home / 'nacre' / 'memory.db'} (group hermes-smoke)")


if __name__ == "__main__":
    main()
