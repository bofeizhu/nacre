#!/usr/bin/env python3
"""Capture a golden trace from pinned Python Graphiti + FalkorDB.

Runs a curated episode set through `graphiti.add_episode` with RECORDING
LLM/embedder clients, then freezes: the episode inputs, every model
exchange, the full resulting graph state (all temporal fields, UUIDs
aliased deterministically), and retrieval results for the set's queries.
The Rust stack (nacre + grit) must reproduce the frozen outputs from the
same episode inputs + recordings.

Usage (manual, networked — never CI):
    docker compose up -d                       # FalkorDB on :6379
    export OPENAI_API_KEY=...                  # LLM + embeddings, or:
    #   CAPTURE_LLM_BASE_URL=https://api.deepseek.com \
    #   CAPTURE_LLM_API_KEY=... CAPTURE_LLM_MODEL=<deepseek model> \
    #   OPENAI_API_KEY=...                     # embeddings only (DeepSeek has none)
    uv run python capture.py episodes/trace1.json
    uv run python capture.py episodes/trace1.json --replay   # offline verify

--replay reruns the pipeline against the saved recordings (no network) and
recaptures the outputs; a diff against the saved trace proves the capture
is deterministic modulo created_at wall-clock fields.

WARNING: capture clears the FalkorDB database it points at. Use the
dedicated docker-compose instance, never a shared one.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

from graphiti_core import Graphiti
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.edges import EntityEdge
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.utils.maintenance.graph_data_operations import clear_data
from recording_clients import (
    FailLoudCrossEncoder,
    RecordingEmbedder,
    RecordingLLMClient,
    RecordingLog,
    ReplayEmbedderClient,
    ReplayLLMClient,
)

ORACLE = Path(__file__).resolve().parent


def build_capture_llm_client():
    """LLM provider from env. Defaults to OpenAI; any OpenAI-compatible
    endpoint works via CAPTURE_LLM_BASE_URL (DeepSeek, vLLM, Together...).
    Custom endpoints use OpenAIGenericClient with json_object structured
    output by default (DeepSeek does not support json_schema); override
    with CAPTURE_LLM_STRUCTURED_OUTPUT=json_schema where supported."""
    base_url = os.environ.get('CAPTURE_LLM_BASE_URL')
    config = LLMConfig(
        api_key=os.environ.get('CAPTURE_LLM_API_KEY') or os.environ.get('OPENAI_API_KEY'),
        base_url=base_url,
        model=os.environ.get('CAPTURE_LLM_MODEL'),
        small_model=os.environ.get('CAPTURE_LLM_SMALL_MODEL')
        or os.environ.get('CAPTURE_LLM_MODEL'),
    )
    if base_url:
        mode = os.environ.get('CAPTURE_LLM_STRUCTURED_OUTPUT', 'json_object')
        return OpenAIGenericClient(config=config, structured_output_mode=mode)
    return OpenAIClient(config=config)


def build_capture_embedder():
    """Embeddings provider from env. DeepSeek has no embeddings endpoint,
    so this typically stays OpenAI (text-embedding-3-small; the whole
    capture costs fractions of a cent) even when the LLM lives elsewhere."""
    kwargs = {
        'api_key': os.environ.get('CAPTURE_EMBEDDER_API_KEY')
        or os.environ.get('OPENAI_API_KEY'),
    }
    if os.environ.get('CAPTURE_EMBEDDER_BASE_URL'):
        kwargs['base_url'] = os.environ['CAPTURE_EMBEDDER_BASE_URL']
    if os.environ.get('CAPTURE_EMBEDDER_MODEL'):
        kwargs['embedding_model'] = os.environ['CAPTURE_EMBEDDER_MODEL']
    return OpenAIEmbedder(config=OpenAIEmbedderConfig(**kwargs))


def iso(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def build_aliases(nodes, edges, episodes) -> dict[str, str]:
    """Deterministic UUID -> alias mapping: run-independent identity.

    Nodes sort by (name, uuid-free tiebreak = summary), episodes by
    (valid_at, name), edges by (fact, endpoints). Collisions fall back to
    insertion order, which is itself deterministic under replay.
    """
    aliases: dict[str, str] = {}
    for i, node in enumerate(sorted(nodes, key=lambda n: (n.name, n.summary or ''))):
        aliases[node.uuid] = f'n{i}'
    for i, ep in enumerate(sorted(episodes, key=lambda e: (iso(e.valid_at) or '', e.name))):
        aliases[ep.uuid] = f'ep{i}'
    for i, edge in enumerate(
        sorted(
            edges,
            # Tiebreak by endpoint ALIASES, not raw uuids — aliases are
            # content-derived, so the ordering is identical across runs and
            # across the Python/Rust implementations.
            key=lambda e: (e.fact, aliases[e.source_node_uuid], aliases[e.target_node_uuid]),
        )
    ):
        aliases[edge.uuid] = f'e{i}'
    return aliases


def dump_graph_state(nodes, edges, episodes, aliases) -> dict:
    def alias(uuid: str) -> str:
        return aliases.get(uuid, f'UNALIASED:{uuid}')

    return {
        'nodes': sorted(
            (
                {
                    'uuid': alias(n.uuid),
                    'name': n.name,
                    'labels': sorted(n.labels or []),
                    'summary': n.summary,
                    'attributes': n.attributes or {},
                    'created_at': iso(n.created_at),
                }
                for n in nodes
            ),
            key=lambda d: d['uuid'],
        ),
        'edges': sorted(
            (
                {
                    'uuid': alias(e.uuid),
                    'source': alias(e.source_node_uuid),
                    'target': alias(e.target_node_uuid),
                    'name': e.name,
                    'fact': e.fact,
                    'episodes': sorted(alias(u) for u in (e.episodes or [])),
                    'attributes': e.attributes or {},
                    'created_at': iso(e.created_at),
                    'valid_at': iso(e.valid_at),
                    'invalid_at': iso(e.invalid_at),
                    'expired_at': iso(e.expired_at),
                }
                for e in edges
            ),
            key=lambda d: d['uuid'],
        ),
        'episodes': sorted(
            (
                {
                    'uuid': alias(ep.uuid),
                    'name': ep.name,
                    'source': ep.source.value,
                    'source_description': ep.source_description,
                    'content': ep.content,
                    'valid_at': iso(ep.valid_at),
                    'created_at': iso(ep.created_at),
                    'entity_edges': sorted(alias(u) for u in (ep.entity_edges or [])),
                }
                for ep in episodes
            ),
            key=lambda d: d['uuid'],
        ),
    }


async def run(spec_path: Path, out_dir: Path, replay: bool) -> None:
    spec = json.loads(spec_path.read_text())
    group_id = spec['group_id']

    driver = FalkorDriver(
        host=os.environ.get('FALKORDB_HOST', 'localhost'),
        port=int(os.environ.get('FALKORDB_PORT', '6379')),
    )

    llm_log = RecordingLog()
    embed_log = RecordingLog()
    if replay:
        llm_client = ReplayLLMClient(RecordingLog.load_index(out_dir / 'llm_recordings.json'))
        embed_index = RecordingLog.load_index(out_dir / 'embedder_recordings.json')
        embed_recordings = json.loads((out_dir / 'embedder_recordings.json').read_text())
        model_id = embed_recordings[0]['request']['model_id'] if embed_recordings else 'unknown'
        embedder = ReplayEmbedderClient(embed_index, model_id)
    else:
        llm_client = RecordingLLMClient(build_capture_llm_client(), llm_log)
        embedder = RecordingEmbedder(build_capture_embedder(), embed_log)

    graphiti = Graphiti(
        graph_driver=driver,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=FailLoudCrossEncoder(),
    )

    await clear_data(driver)
    await graphiti.build_indices_and_constraints()

    for episode in spec['episodes']:
        await graphiti.add_episode(
            name=episode['name'],
            episode_body=episode['content'],
            source_description=episode['source_description'],
            reference_time=datetime.fromisoformat(episode['reference_time']),
            source=EpisodeType[episode.get('source', 'message')],
            group_id=group_id,
        )
        print(f'  added {episode["name"]}')

    nodes = await EntityNode.get_by_group_ids(driver, [group_id])
    edges = await EntityEdge.get_by_group_ids(driver, [group_id])
    episodes = await EpisodicNode.get_by_group_ids(driver, [group_id])
    aliases = build_aliases(nodes, edges, episodes)
    graph_state = dump_graph_state(nodes, edges, episodes, aliases)

    retrieval = []
    for query in spec.get('queries', []):
        hits = await graphiti.search(query, group_ids=[group_id], num_results=10)
        retrieval.append(
            {
                'query': query,
                'results': [
                    {
                        'uuid': aliases.get(h.uuid, f'UNALIASED:{h.uuid}'),
                        'fact': h.fact,
                        'valid_at': iso(h.valid_at),
                        'invalid_at': iso(h.invalid_at),
                    }
                    for h in hits
                ],
            }
        )

    suffix = '.replay' if replay else ''
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f'graph_state{suffix}.json').write_text(
        json.dumps(graph_state, ensure_ascii=False, indent=1, sort_keys=True) + '\n'
    )
    (out_dir / f'retrieval{suffix}.json').write_text(
        json.dumps(retrieval, ensure_ascii=False, indent=1, sort_keys=True) + '\n'
    )
    if not replay:
        (out_dir / 'episodes.json').write_text(
            json.dumps(spec, ensure_ascii=False, indent=1, sort_keys=True) + '\n'
        )
        llm_log.save(out_dir / 'llm_recordings.json')
        embed_log.save(out_dir / 'embedder_recordings.json')
        meta = {
            'graphiti_core': '0.29.2',
            'trace': spec['name'],
            'llm_exchanges': len(llm_log.recordings),
            'embedder_exchanges': len(embed_log.recordings),
            'note': 'record FalkorDB image digest + capture date here at capture time',
        }
        (out_dir / 'meta.json').write_text(
            json.dumps(meta, ensure_ascii=False, indent=1, sort_keys=True) + '\n'
        )
    print(f'wrote trace to {out_dir}{" (replay outputs)" if replay else ""}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episodes', type=Path, help='episode-set spec JSON')
    parser.add_argument('--out', type=Path, default=None, help='trace output dir')
    parser.add_argument('--replay', action='store_true', help='replay saved recordings (offline)')
    args = parser.parse_args()

    out_dir = args.out or (ORACLE / 'fixtures' / args.episodes.stem)
    asyncio.run(run(args.episodes, out_dir, args.replay))


if __name__ == '__main__':
    main()
