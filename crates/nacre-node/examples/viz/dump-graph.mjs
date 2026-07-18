// Dumps a smoke memory graph as JSON for the viewer — the data-contract
// proof for the Layer 3 graph view. Replay-ingests the committed golden
// trace (oracle/fixtures/trace1) through the built addon (offline, no
// network), then dumps every read-path row the visualization needs:
// nodes (labels, bi-temporal fields), edges (validity, provenance
// episode ids via mentionsOf), and episodes.
//
// Usage: node examples/viz/dump-graph.mjs                    (after `npm run build`)
//        node examples/viz/dump-graph.mjs --db <path> --group <id>
//
// Default: replay-ingest trace1 into a temp db and dump it. With --db,
// dump an EXISTING graph instead (e.g. one captured by the Hermes
// provider) — no ingestion, no recordings needed.
//
// Writes, next to this script:
//   graph.json     — the pure data contract (committed as a sample)
//   graph.data.js  — the same object as `window.NACRE_GRAPH = …`, so
//                    viewer.html works from file:// without a server
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(here, '..', '..');
const { Memory } = await import(path.join(root, 'index.js'));

const args = process.argv.slice(2);
const argOf = (flag) => {
  const i = args.indexOf(flag);
  return i >= 0 ? args[i + 1] : undefined;
};
const existingDb = argOf('--db');
const existingGroup = argOf('--group');
if (existingDb && !existingGroup) {
  console.error('--db requires --group <group id>');
  process.exit(1);
}

if (existingDb) {
  const m = Memory.open(existingDb, 'viz-dump');
  writeGraph(m, existingGroup, `existing graph (${existingDb})`);
  process.exit(0);
}

const oracle = path.join(root, '..', '..', 'oracle');
const fixtures = path.join(oracle, 'fixtures', 'trace1');
const spec = JSON.parse(fs.readFileSync(path.join(oracle, 'episodes', 'trace1.json'), 'utf8'));
const replayLlm = { provider: 'replay', recordingsPath: path.join(fixtures, 'llm_recordings.json') };
const replayEmb = { provider: 'replay', recordingsPath: path.join(fixtures, 'embedder_recordings.json') };

const dbPath = path.join(os.tmpdir(), `nacre-viz-dump-${process.pid}.db`);
try { fs.unlinkSync(dbPath); } catch {}
const m = Memory.open(dbPath, 'viz-dump');

for (const ep of spec.episodes) {
  const out = await m.addEpisode(
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
  console.log(
    `${ep.name}: +${out.nodeIds.length} nodes, +${out.newEdgeIds.length} edges, ` +
      `${out.merges.length} merges, ${out.invalidatedEdgeIds.length} invalidated`,
  );
}

writeGraph(m, spec.group_id, 'trace1 replay (oracle/fixtures/trace1)');
fs.unlinkSync(dbPath);

// Full bi-temporal rows — the viewer decides what to show (live view vs
// audit view). Edge provenance comes from mentionsOf, per the contract.
function writeGraph(memory, groupId, sourceLabel) {
  const graph = {
    groupId,
    source: sourceLabel,
    nodes: memory.nodesInGroup(groupId),
    edges: memory.edgesInGroup(groupId).map((e) => ({ ...e, episodes: memory.mentionsOf(e.id) })),
    episodes: memory.episodesInGroup(groupId),
  };
  const json = JSON.stringify(graph, null, 1);
  fs.writeFileSync(path.join(here, 'graph.json'), json + '\n');
  fs.writeFileSync(path.join(here, 'graph.data.js'), `window.NACRE_GRAPH = ${json};\n`);
  const live = graph.nodes.filter((n) => !n.expiredAt).length;
  console.log(
    `\nwrote graph.json + graph.data.js: ${graph.nodes.length} nodes (${live} live), ` +
      `${graph.edges.length} edges, ${graph.episodes.length} episodes`,
  );
  console.log('open viewer.html in a browser to see it');
}
