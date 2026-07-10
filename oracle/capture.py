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

from dotenv import load_dotenv

# Graphiti bulk-saves nodes/edges concurrently (semaphore_gather); FalkorDB
# insertion order then varies run-to-run, which changes internal-id tie-break
# order in fulltext/vector search — and therefore the ORDER of candidate lists
# embedded in prompts. Serializing everything makes capture and replay execute
# the identical sequence, which is what makes traces byte-deterministic.
# Must be set before graphiti_core imports (read at import time).
os.environ['SEMAPHORE_LIMIT'] = '1'

from graphiti_core import Graphiti
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.edges import EntityEdge
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.utils.maintenance.graph_data_operations import clear_data
import graphiti_core.utils.maintenance.edge_operations as _edge_ops

# ORACLE DEVIATION (see DEVIATIONS.md "Edge dedup/invalidation candidate
# pools"): upstream builds the EXISTING FACTS / FACT INVALIDATION CANDIDATES
# prompt lists from a FalkorDB hybrid search — engine-internal result order
# (collect() ignores ORDER BY) and top-10 relevance truncation are both
# engine-ranking behavior that neither a rerun of this harness nor any other
# storage engine can reproduce. Replace the edge-resolution search with an
# engine-free equivalent: ALL saved group edges, restricted to the caller's
# edge_uuids filter when present (that filter carries the same-endpoint-pair
# edges, so the related/invalidation split is preserved), sorted by fact
# text. nacre builds its pools identically.


async def _deterministic_edge_search(clients, query, group_ids, config=None,
                                     search_filter=None, *args, **kwargs):
    from graphiti_core.search.search_config import SearchResults

    group_id = group_ids[0]
    dump_driver = clients.driver.clone(database=group_id)
    try:
        edges = await EntityEdge.get_by_group_ids(dump_driver, [group_id])
    except Exception:  # raised when the group has no edges yet
        edges = []
    allowed = getattr(search_filter, 'edge_uuids', None) if search_filter is not None else None
    if allowed is not None:
        allowed_set = set(allowed)
        edges = [e for e in edges if e.uuid in allowed_set]
    edges.sort(key=lambda e: (e.fact, e.uuid))
    return SearchResults(edges=edges)


_edge_ops.search = _deterministic_edge_search

# ORACLE DEVIATION (see DEVIATIONS.md "Node dedup candidate search"):
# upstream ranks node dedup candidates with an in-engine vector search
# (same nondeterminism class as the edge search above). Replace it with an
# engine-free equivalent both sides of the oracle can compute bit-for-bit:
# embed the extracted names and every distinct existing name (sorted, so
# capture and nacre issue identical — recordable — embedder requests), then
# rank by cosine over f32-truncated components with sequential f64
# accumulation (IEEE-identical to nacre's cosine_f64), strict > min score,
# limit 15. The uuid tie-break is per-run, but score ties require identical
# names, which the exact-match fast path resolves before any prompt.
import graphiti_core.utils.maintenance.node_operations as _node_ops
import math as _math
import struct as _struct


def _f32(x: float) -> float:
    return _struct.unpack('f', _struct.pack('f', x))[0]


def _cosine_f64(a: list[float], b: list[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        na += x * x
        nb += y * y
    return dot / (_math.sqrt(na) * _math.sqrt(nb))


async def _deterministic_node_candidates(clients, extracted_nodes):
    if not extracted_nodes:
        return []
    group_id = extracted_nodes[0].group_id
    dump_driver = clients.driver.clone(database=group_id)
    try:
        existing = await EntityNode.get_by_group_ids(dump_driver, [group_id])
    except Exception:  # raised when the group has no nodes yet
        existing = []
    queries = [n.name.replace('\n', ' ') for n in extracted_nodes]
    query_vectors = await clients.embedder.create_batch(queries)
    if not existing:
        return [[] for _ in extracted_nodes]
    names = sorted({n.name for n in existing})
    name_vectors = {
        name: [_f32(x) for x in vec]
        for name, vec in zip(names, await clients.embedder.create_batch(names), strict=True)
    }
    results = []
    for query_vector in query_vectors:
        qv = [_f32(x) for x in query_vector]
        scored = []
        for node in existing:
            score = _cosine_f64(qv, name_vectors[node.name])
            if score > _node_ops.NODE_DEDUP_COSINE_MIN_SCORE:
                scored.append((score, node.uuid, node))
        scored.sort(key=lambda t: (-t[0], t[1]))
        results.append([n for _, _, n in scored[: _node_ops.NODE_DEDUP_CANDIDATE_LIMIT]])
    return results


_node_ops._semantic_candidate_search = _deterministic_node_candidates




from recording_clients import (
    FailLoudCrossEncoder,
    RecordingEmbedder,
    RecordingLLMClient,
    RecordingLog,
    ReplayEmbedderClient,
    ReplayLLMClient,
)

ORACLE = Path(__file__).resolve().parent

# Keys live in oracle/.env (gitignored); shell env vars take precedence.
load_dotenv(ORACLE / '.env')


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

    # clear_data on the base driver only touches the default graph;
    # FalkorDB shards one graph per group_id, so clear that too (a crashed
    # or prior run would otherwise contaminate this capture's contexts).
    await clear_data(driver)
    await clear_data(driver.clone(database=group_id))
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

    # FalkorDB shards one graph per group_id — clone the driver onto it.
    dump_driver = driver.clone(database=group_id)
    nodes = await EntityNode.get_by_group_ids(dump_driver, [group_id])
    edges = await EntityEdge.get_by_group_ids(dump_driver, [group_id])
    episodes = await EpisodicNode.get_by_group_ids(dump_driver, [group_id])
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

    if replay:
        _replay_verdict(out_dir)


def _normalize_state(obj):
    """created_at/expired_at are wall-clock: drop / reduce to presence."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == 'created_at':
                continue
            if k == 'expired_at':
                out[k] = v is not None
            else:
                out[k] = _normalize_state(v)
        return out
    if isinstance(obj, list):
        return [_normalize_state(x) for x in obj]
    return obj


def _replay_verdict(out_dir: Path) -> None:
    """Fail loudly unless the replay reproduced the captured graph state.

    Retrieval is compared informationally only: FalkorDB's HNSW vector index
    is nondeterministic across processes, so rank order (and top-k membership
    at the tail) is not reproducible even by Graphiti itself. The oracle's
    retrieval leg is therefore advisory — see DEVIATIONS.md.
    """
    captured = json.loads((out_dir / 'graph_state.json').read_text())
    replayed = json.loads((out_dir / 'graph_state.replay.json').read_text())
    if _normalize_state(captured) != _normalize_state(replayed):
        raise SystemExit(
            'REPLAY FAILED: graph state diverges from capture '
            '(diff graph_state.json vs graph_state.replay.json)'
        )
    print('replay verdict: graph state DETERMINISTIC (modulo uuids/wall-clock)')
    r_cap = json.loads((out_dir / 'retrieval.json').read_text())
    r_rep = json.loads((out_dir / 'retrieval.replay.json').read_text())
    for qc, qr in zip(r_cap, r_rep):
        fc = [h['fact'] for h in qc['results']]
        fr = [h['fact'] for h in qr['results']]
        status = (
            'rank-identical'
            if fc == fr
            else ('same set, rank differs' if sorted(fc) == sorted(fr) else 'SET DIFFERS')
        )
        print(f'  retrieval [{status}]: {qc["query"]}')


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
