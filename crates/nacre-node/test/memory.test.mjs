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
let EPISODE_EXISTS_CODE;
let EPISODE_NOT_FOUND_CODE;
try {
  ({ Memory, EPISODE_EXISTS_CODE, EPISODE_NOT_FOUND_CODE } = await import(
    path.join(root, 'index.js')
  ));
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

function ingest(m, ep, { id, groupId, metadata } = {}) {
  return m.addEpisode(
    {
      id,
      name: ep.name,
      content: ep.content,
      sourceDescription: ep.source_description,
      groupId: groupId ?? spec.group_id,
      validAt: ep.reference_time,
      metadata,
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

test('a caller-supplied id makes ingestion idempotent', { skip: !Memory }, async () => {
  // Its own file: two tests sharing one database path collide.
  const dbPath = path.join(os.tmpdir(), `nacre-node-idempotent-${process.pid}.db`);
  try { fs.unlinkSync(dbPath); } catch {}
  const m = Memory.open(dbPath, 'node-test');

  // A caller that retries failed ingests owns the id — the same one on
  // every attempt for the same episode.
  const id = '3f2504e0-4f89-11d3-9a0c-0305e82c3301';
  const first = await ingest(m, spec.episodes[0], { id });
  assert.equal(first.episodeId, id, 'the supplied id IS the stored episode id');
  const nodes = m.nodesInGroup(spec.group_id).length;

  // The retry that arrives after the first attempt actually landed is
  // refused, and refused BEFORE any work: the episode row commits with the
  // graph mutations it evidences, so its presence means the whole ingest
  // is already there.
  await assert.rejects(
    () => ingest(m, spec.episodes[0], { id }),
    (err) => {
      assert.equal(
        err.message.split(': ')[0],
        EPISODE_EXISTS_CODE,
        `the rejection leads with the stable code: ${err.message}`,
      );
      assert.ok(err.message.includes(id), err.message);
      return true;
    },
  );
  assert.equal(m.episodesInGroup(spec.group_id).length, 1, 'no second episode row');
  assert.equal(m.nodesInGroup(spec.group_id).length, nodes, 'no second set of entities');

  // A malformed key is a caller bug, caught with the argument errors.
  await assert.rejects(
    () => ingest(m, spec.episodes[0], { id: 'not-a-uuid' }),
    (err) => {
      assert.equal(err.code, 'InvalidArg');
      assert.match(err.message, /^id "not-a-uuid":/);
      return true;
    },
  );

  fs.unlinkSync(dbPath);
});

test('forgetting and episode metadata through the FFI (replay, offline)', { skip: !Memory }, async () => {
  // Its own file, like every other test here.
  const dbPath = path.join(os.tmpdir(), `nacre-node-forget-${process.pid}.db`);
  try { fs.unlinkSync(dbPath); } catch {}
  const m = Memory.open(dbPath, 'node-test');
  const sibling = `${spec.group_id}-sibling`;

  // The same trace into two groups of one database. The replay key carries
  // no group id while every candidate pool the pipeline consults IS
  // group-scoped, so the second ingest issues byte-identical requests and
  // builds a structurally identical, id-disjoint sibling — the control that
  // deleting one group must leave untouched.
  const provenance = JSON.stringify({ outboxRow: 42, consent: 'granted' });
  const first = await ingest(m, spec.episodes[0], { metadata: provenance });
  const second = await ingest(m, spec.episodes[1]);
  for (const ep of spec.episodes) await ingest(m, ep, { groupId: sibling });

  // --- metadata: stored verbatim, handed back by every episode read path,
  // and never sent to a model (a leak into a prompt would change the
  // replay key and fail this ingest as a miss, not as an assertion). ---
  const stored = m.episode(first.episodeId);
  assert.equal(stored.metadata, provenance, 'metadata round-trips through the getter');
  assert.equal(stored.id, first.episodeId);
  const scanned = m.episodesInGroup(spec.group_id).find((e) => e.id === first.episodeId);
  assert.equal(scanned.metadata, provenance, 'and through the group scan');
  const windowed = m.previousEpisodes(spec.group_id).find((e) => e.id === first.episodeId);
  assert.equal(windowed.metadata, provenance, 'and through the previous-episode window');
  assert.ok(!m.episode(second.episodeId).metadata, 'an episode that stored nothing has none');

  // The getter answers "no such row" with null; only a malformed id is an
  // argument error.
  assert.equal(m.episode('018f0000-0000-7000-8000-000000000000'), null, 'unknown id → null');
  assert.throws(() => m.episode('not-a-uuid'), { code: 'InvalidArg' });

  // --- deleteEpisode: sole evidence goes, corroborated evidence stays ---
  const siblingBefore = {
    nodes: m.nodesInGroup(sibling).length,
    edges: m.edgesInGroup(sibling).length,
    episodes: m.episodesInGroup(sibling).length,
  };
  const gone = m.deleteEpisode(second.episodeId);
  assert.equal(gone.episodeId, second.episodeId);
  assert.ok(gone.purgedNodeIds.length + gone.purgedEdgeIds.length > 0, 'ep-1 left residue');
  assert.ok(gone.keptNodeIds.length > 0, 'ep-0 corroborates entities ep-1 also mentioned');
  assert.equal(m.episode(second.episodeId), null, 'the episode row is unreachable');
  const liveIds = new Set(m.nodesInGroup(spec.group_id).map((n) => n.id));
  for (const id of gone.purgedNodeIds) assert.ok(!liveIds.has(id), `${id} is gone`);
  for (const id of gone.keptNodeIds) assert.ok(liveIds.has(id), `${id} survived`);

  // A second delete is refused with the stable prefix, and writes nothing.
  assert.throws(
    () => m.deleteEpisode(second.episodeId),
    (err) => {
      assert.equal(
        err.message.split(': ')[0],
        EPISODE_NOT_FOUND_CODE,
        `the rejection leads with the stable code: ${err.message}`,
      );
      assert.ok(err.message.includes(second.episodeId), err.message);
      return true;
    },
  );
  assert.throws(() => m.deleteEpisode('not-a-uuid'), { code: 'InvalidArg' });

  // --- deleteGroup: one namespace, whole; the sibling untouched ---
  const wiped = m.deleteGroup(spec.group_id);
  assert.ok(wiped.purgedEpisodeIds.includes(first.episodeId));
  assert.equal(m.nodesInGroup(spec.group_id).length, 0, 'no node rows left');
  assert.equal(m.edgesInGroup(spec.group_id).length, 0, 'no edge rows left');
  assert.equal(m.episodesInGroup(spec.group_id).length, 0, 'no episode rows left');
  assert.equal(m.episode(first.episodeId), null, 'metadata went with its episode');
  assert.deepEqual(
    {
      nodes: m.nodesInGroup(sibling).length,
      edges: m.edgesInGroup(sibling).length,
      episodes: m.episodesInGroup(sibling).length,
    },
    siblingBefore,
    'the sibling group is untouched by either deletion',
  );

  // The FTS mirrors and vector tables are SHARED across groups — a delete
  // that was not group-scoped would show up here, as the sibling losing
  // recall or the wiped group keeping it.
  assert.deepEqual(
    await m.searchEdges(spec.queries[0], spec.group_id, 3, replayEmb),
    [],
    'a forgotten group is unreachable through search',
  );
  assert.ok(
    (await m.searchEdges(spec.queries[0], sibling, 3, replayEmb)).length >= 1,
    'while the sibling still recalls the same fact',
  );

  // An unknown group is a group with no rows, not an error.
  assert.deepEqual(m.deleteGroup('never-existed'), {
    purgedNodeIds: [],
    purgedEdgeIds: [],
    purgedEpisodeIds: [],
  });

  fs.unlinkSync(dbPath);
});
