// Offline test of the sidecar process: spawns the real child, speaks the
// ndjson protocol over stdio, replays the committed trace1 recordings —
// no network, ever. Skips loudly when the addon has not been built.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import readline from 'node:readline';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(here, '..');
const addonBuilt = fs.existsSync(path.join(root, 'index.js'));
if (!addonBuilt) {
  console.warn('SKIP: nacre-node addon not built — run `npm run build` first');
}

const fixtures = path.join(root, '..', '..', 'oracle', 'fixtures', 'trace1');
const spec = JSON.parse(
  fs.readFileSync(path.join(root, '..', '..', 'oracle', 'episodes', 'trace1.json'), 'utf8'),
);

/** Minimal ndjson client over a spawned sidecar. */
function startSidecar(env) {
  const child = spawn(process.execPath, [path.join(root, 'sidecar', 'sidecar.mjs')], {
    stdio: ['pipe', 'pipe', 'inherit'],
    env: { ...process.env, ...env },
  });
  const pending = new Map();
  let nextId = 1;
  const rl = readline.createInterface({ input: child.stdout });
  rl.on('line', (line) => {
    const msg = JSON.parse(line);
    const entry = pending.get(msg.id);
    if (!entry) throw new Error(`response for unknown id: ${line}`);
    pending.delete(msg.id);
    entry(msg);
  });
  return {
    child,
    call(method, params) {
      const id = nextId++;
      return new Promise((resolve) => {
        pending.set(id, resolve);
        child.stdin.write(JSON.stringify({ id, method, params }) + '\n');
      });
    },
    exited: new Promise((resolve) => child.on('exit', resolve)),
  };
}

test('sidecar: init → addEpisode → searchEdges → episode → delete → status → shutdown (replay, offline)', { skip: !addonBuilt }, async () => {
  const dbPath = path.join(os.tmpdir(), `nacre-sidecar-test-${process.pid}.db`);
  try { fs.unlinkSync(dbPath); } catch {}
  const sc = startSidecar({
    NACRE_LLM_PROVIDER: 'replay',
    NACRE_LLM_RECORDINGS: path.join(fixtures, 'llm_recordings.json'),
    NACRE_EMBEDDER_PROVIDER: 'replay',
    NACRE_EMBEDDER_RECORDINGS: path.join(fixtures, 'embedder_recordings.json'),
  });

  // Pre-init calls fail loudly.
  const early = await sc.call('addEpisode', { content: 'x', sourceDescription: 'y' });
  assert.match(early.error, /not initialized/);

  const init = await sc.call('init', {
    dbPath,
    deviceId: 'sidecar-test',
    groupId: spec.group_id,
  });
  assert.ok(init.result.version.length > 0);
  assert.equal(init.result.groupId, spec.group_id);

  // Write path: two trace episodes, deterministic deltas under replay.
  // ep-0 also carries the caller's own two fields — the idempotency key and
  // an opaque metadata payload — which the sidecar must forward verbatim.
  const keyedId = '3f2504e0-4f89-11d3-9a0c-0305e82c3302';
  const provenance = JSON.stringify({ outboxRow: 7 });
  const ep = (i, extra = {}) =>
    sc.call('addEpisode', {
      name: spec.episodes[i].name,
      content: spec.episodes[i].content,
      source: spec.episodes[i].source,
      sourceDescription: spec.episodes[i].source_description,
      validAt: spec.episodes[i].reference_time,
      ...extra,
    });
  const first = await ep(0, { id: keyedId, metadata: provenance });
  assert.equal(first.result.nodeIds.length, 4, 'ep-0 extracts 4 entities');
  assert.equal(first.result.newEdgeIds.length, 2, 'ep-0 creates 2 facts');
  assert.equal(first.result.episodeId, keyedId, 'the supplied id is forwarded, not minted over');
  const second = await ep(1);
  assert.ok(second.result.merges.length > 0, 'ep-1 dedups returning entities');

  // A retry under the stored id is refused by the pipeline, and the code
  // prefix survives the protocol's error rendering.
  const retry = await ep(0, { id: keyedId, metadata: provenance });
  assert.match(retry.error, /^EPISODE_EXISTS: /);

  // Search with the recorded query embedding; hits carry provenance.
  const search = await sc.call('searchEdges', { query: spec.queries[0], limit: 3 });
  assert.ok(search.result.hits.length >= 1, 'recorded query returns hits');
  assert.ok(search.result.hits[0].fact.length > 0);
  assert.ok(search.result.hits[0].episodes.length >= 1);

  // Error surfaces are responses, not crashes.
  const miss = await sc.call('searchEdges', { query: 'an unrecorded query' });
  assert.match(miss.error, /no recording matches/);
  const unknown = await sc.call('nope', {});
  assert.match(unknown.error, /unknown method/);

  const status = await sc.call('status', {});
  assert.equal(status.result.episodes, 2);
  assert.ok(status.result.liveNodes >= 4);
  assert.ok(status.result.edges >= 2);

  // Provenance drill-down: the ids searchEdges hands back are readable as
  // rows, metadata included. An unknown id is null, not an error.
  const lookup = await sc.call('episode', { id: keyedId });
  assert.equal(lookup.result.episode.metadata, provenance, 'metadata survives the round trip');
  assert.equal(lookup.result.episode.id, keyedId);
  const missing = await sc.call('episode', { id: '018f0000-0000-7000-8000-000000000000' });
  assert.equal(missing.result.episode, null);

  // Forgetting: one episode, then the whole group.
  const forgotten = await sc.call('deleteEpisode', { id: second.result.episodeId });
  assert.equal(forgotten.result.episodeId, second.result.episodeId);
  assert.ok(forgotten.result.keptNodeIds.length > 0, 'ep-0 corroborates entities ep-1 shared');
  const twice = await sc.call('deleteEpisode', { id: second.result.episodeId });
  assert.match(twice.error, /^EPISODE_NOT_FOUND: /, 'the code prefix reaches the protocol');

  const wiped = await sc.call('deleteGroup', {});
  assert.ok(wiped.result.purgedEpisodeIds.includes(keyedId), 'groupId defaults to the init one');
  const empty = await sc.call('status', {});
  assert.equal(empty.result.episodes, 0);
  assert.equal(empty.result.liveNodes, 0);
  assert.equal(empty.result.edges, 0);

  const bye = await sc.call('shutdown', {});
  assert.equal(bye.result.ok, true);
  assert.equal(await sc.exited, 0, 'clean exit after shutdown');
  fs.unlinkSync(dbPath);
});

test('sidecar: missing credentials fail init loudly (offline)', { skip: !addonBuilt }, async () => {
  const sc = startSidecar({
    NACRE_LLM_PROVIDER: 'deepseek',
    NACRE_LLM_API_KEY: '',
    NACRE_EMBEDDER_PROVIDER: 'replay',
    NACRE_EMBEDDER_RECORDINGS: '/nonexistent',
  });
  const init = await sc.call('init', {
    dbPath: path.join(os.tmpdir(), `nacre-sidecar-noinit-${process.pid}.db`),
    deviceId: 'x',
    groupId: 'g',
  });
  assert.match(init.error, /NACRE_LLM_API_KEY/);
  sc.child.stdin.end();
  assert.equal(await sc.exited, 0, 'exits when parent closes stdin');
});
