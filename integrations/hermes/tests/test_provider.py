"""Offline tests of the capture-only provider: real sidecar, replay mode."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TRACE1 = json.loads((REPO_ROOT / "oracle" / "episodes" / "trace1.json").read_text())


def _sequenced_clock(times):
    """Clock returning the trace's reference times, in order."""
    it = iter(times)
    last = times[-1]
    return lambda: datetime.fromisoformat(next(it, last))


def _raw_content_provider(plugin, clock):
    """Provider whose episode formatting passes user_content through raw —
    lets the test feed trace1 episode content verbatim so the replay
    recordings match. The default formatting has its own unit test."""

    class RawProvider(plugin.NacreMemoryProvider):
        def _format_episode(self, user_content, assistant_content):
            return user_content

    return RawProvider(clock=clock)


def _init(provider, tmp_path, group_id, **kwargs):
    (tmp_path / "nacre.json").write_text(json.dumps({"group_id": group_id}))
    provider.initialize(
        "test-session",
        hermes_home=str(tmp_path),
        platform="cli",
        agent_context="primary",
        **kwargs,
    )
    return provider


def test_capture_replay_end_to_end(plugin, replay_env, tmp_path):
    times = [ep["reference_time"] for ep in TRACE1["episodes"]]
    p = _raw_content_provider(plugin, _sequenced_clock(times))
    assert p.is_available(), "replay env + built addon should be available"
    _init(p, tmp_path, TRACE1["group_id"])
    try:
        assert p._active, "provider should be active after initialize"

        p.sync_turn(TRACE1["episodes"][0]["content"], "")
        p.flush()
        assert p.last_result is not None, "first episode ingested"
        assert len(p.last_result["nodeIds"]) == 4, "ep-0 extracts 4 entities"
        assert len(p.last_result["newEdgeIds"]) == 2

        p.sync_turn(TRACE1["episodes"][1]["content"], "")
        p.flush()
        assert len(p.last_result["merges"]) > 0, "ep-1 dedups returning entities"

        stats = p.stats()
        assert stats["episodes"] == 2
        assert stats["liveNodes"] >= 4
        assert stats["groupId"] == TRACE1["group_id"]

        # Recall works at the sidecar level (Stage 2 material) even though
        # the provider deliberately exposes no prefetch or tools.
        hits = p._client.call(
            "searchEdges", {"query": TRACE1["queries"][0], "limit": 3}
        )["hits"]
        assert len(hits) >= 1 and hits[0]["episodes"]

        # The db landed under hermes_home (covered by `hermes backup`).
        assert (tmp_path / "nacre" / "memory.db").exists()
        assert p.backup_paths() == []
    finally:
        p.shutdown()
    assert not p._active


def test_breaker_opens_and_never_raises(plugin, replay_env, tmp_path):
    p = _raw_content_provider(plugin, _sequenced_clock(["2026-01-01T00:00:00+00:00"]))
    _init(p, tmp_path, "breaker-test")
    try:
        # Content with no recording -> replay fails loudly sidecar-side;
        # the provider absorbs it. Three strikes open the breaker.
        for i in range(3):
            p.sync_turn(f"unrecorded content {i}", "")
        p.flush()
        assert p._breaker_open, "breaker opens after 3 consecutive failures"
        assert not p._active
        p.sync_turn("after breaker", "")  # must be a silent no-op
        assert p._queue.qsize() == 0
    finally:
        p.shutdown()


def test_non_primary_context_never_spawns(plugin, replay_env, tmp_path):
    p = plugin.NacreMemoryProvider()
    p.initialize(
        "cron-session", hermes_home=str(tmp_path), platform="cron", agent_context="cron"
    )
    assert not p._active
    assert p._client is None, "no sidecar for non-primary contexts"
    p.sync_turn("cron output", "should be ignored")
    assert p._queue.qsize() == 0
    p.shutdown()


def test_stage1_surface_is_invisible(plugin):
    p = plugin.NacreMemoryProvider()
    assert p.name == "nacre"
    assert p.system_prompt_block() == ""
    assert p.get_tool_schemas() == []
    assert p.prefetch("anything") == ""


def test_default_episode_format(plugin):
    p = plugin.NacreMemoryProvider()
    p._config = {}
    assert p._format_episode("hi", "hello") == "user: hi\nassistant: hello"
    p._config = {"user_label": "Bofei", "assistant_label": "Hermes"}
    assert p._format_episode("hi", "yo") == "Bofei: hi\nHermes: yo"


def test_config_schema_contract(plugin, tmp_path):
    p = plugin.NacreMemoryProvider()
    schema = p.get_config_schema()
    keys = {f["key"] for f in schema}
    assert {"llm_api_key", "embedder_api_key", "group_id"} <= keys
    secrets = [f for f in schema if f.get("secret")]
    assert all(f.get("env_var", "").startswith("NACRE_") for f in secrets)
    # save_config writes non-secret values, never echoes secrets back.
    p.save_config({"group_id": "custom", "nacre_node_dir": ""}, str(tmp_path))
    saved = json.loads((tmp_path / "nacre.json").read_text())
    assert saved == {"group_id": "custom"}, "empty values dropped, no secrets"
