// Offline, deterministic Node-side test of the bindings: replays the
// committed golden-trace recordings (oracle/fixtures/trace1) through the
// full pipeline — no network, ever. Skips (loudly) when the addon has not
// been built (`npm run build` first); cargo-only CI stays green.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(here, '..');

let Memory;
try {
  ({ Memory } = await import(path.join(root, 'index.js')));
} catch {
  console.warn('SKIP: nacre-node addon not built — run `npm run build` first');
}

const fixtures = path.join(root, '..', '..', 'oracle', 'fixtures', 'trace1');
const spec = JSON.parse(
  fs.readFileSync(path.join(root, '..', '..', 'oracle', 'episodes', 'trace1.json'), 'utf8'),
);
const replayLlm = {
  provider: 'replay',
  recordingsPath: path.join(fixtures, 'llm_recordings.json'),
};
const replayEmb = {
  provider: 'replay',
  recordingsPath: path.join(fixtures, 'embedder_recordings.json'),
};

function ingest(m, ep) {
  return m.addEpisode(
    {
      name: ep.name,
      content: ep.content,
      sourceDescription: ep.source_description,
      groupId: spec.group_id,
      validAt: ep.reference_time,
    },
    replayLlm,
    replayEmb,
  );
}

test('write, read, and search through the FFI (replay, offline)', { skip: !Memory }, async () => {
  const dbPath = path.join(os.tmpdir(), `nacre-node-test-${process.pid}.db`);
  try { fs.unlinkSync(dbPath); } catch {}
  const m = Memory.open(dbPath, 'node-test');

  // --- write path: deltas are deterministic under replay ---
  const first = await ingest(m, spec.episodes[0]);
  assert.equal(first.nodeIds.length, 4, 'ep-0 extracts 4 entities');
  assert.equal(first.newEdgeIds.length, 2, 'ep-0 creates 2 facts');
  assert.equal(first.merges.length, 0);
  assert.equal(first.invalidatedEdgeIds.length, 0);

  const second = await ingest(m, spec.episodes[1]);
  assert.ok(second.merges.length > 0, 'ep-1 dedups returning entities');

  // --- read path: the viz data contract ---
  const nodes = m.nodesInGroup(spec.group_id);
  const live = nodes.filter((n) => !n.expiredAt);
  const edges = m.edgesInGroup(spec.group_id);
  const episodes = m.episodesInGroup(spec.group_id);
  assert.equal(episodes.length, 2);
  assert.ok(edges.length >= 4, `edges: ${edges.length}`);
  assert.ok(live.length >= 7, `live nodes: ${live.length}`);
  assert.ok(nodes.length > live.length, 'merged-away drafts retained for audit');
  assert.ok(nodes[0].labels.includes('Entity'));
  assert.match(nodes[0].createdAt, /^\d{4}-\d{2}-\d{2}T.*Z$/);

  const priya = live.find((n) => n.name === 'Priya');
  assert.ok(priya, 'Priya is a live node');
  assert.ok(priya.summary.length > 0, 'summaries persisted');

  const sub = m.traverse([priya.id], { depth: 1, maxNodes: 16 });
  assert.ok(sub.nodes.some((n) => n.id === priya.id));
  assert.ok(sub.edges.length >= 1, 'neighborhood has edges');

  const history = m.nodeHistory(priya.id);
  assert.ok(history.edges.length >= 1, 'audit trail present');

  const mentions = m.mentionsOf(edges[0].id);
  assert.equal(mentions.length, 1, 'fact traces to its episode');
  assert.ok(episodes.some((e) => e.id === mentions[0]));

  const window = m.previousEpisodes(spec.group_id);
  assert.equal(window.length, 2);
  assert.ok(window[0].occurredAt <= window[1].occurredAt, 'chronological');

  // --- search: recorded query embedding, hybrid recall with provenance ---
  const hits = await m.searchEdges(spec.queries[0], spec.group_id, 3, replayEmb);
  assert.ok(hits.length >= 1, 'recorded query returns hits');
  assert.ok(hits[0].fact.length > 0);
  assert.ok(hits[0].episodes.length >= 1, 'hits carry provenance');

  // --- error surfaces fail loudly, not silently ---
  await assert.rejects(
    () => m.searchEdges('an unrecorded query', spec.group_id, 3, replayEmb),
    /no recording matches/,
    'replay refuses to guess',
  );

  fs.unlinkSync(dbPath);
});
