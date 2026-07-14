// Live end-to-end smoke of the bindings: real LLM + real embeddings, no
// recordings. NOT part of `npm test` — run explicitly with keys in env:
//
//   set -a && . ../../oracle/.env && set +a
//   npm run build:live && node scripts/live-smoke.mjs
//   (the default build ships without networked providers; deepseek/zhipu
//   need the live-providers feature)
//
// Env (never printed): CAPTURE_LLM_API_KEY (DeepSeek, Anthropic-style
// endpoint) and CAPTURE_EMBEDDER_API_KEY (Zhipu embedding-3); set
// NACRE_SMOKE_ANTHROPIC_KEY to use Anthropic instead.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const { Memory } = await import(path.join(here, '..', 'index.js'));

const llm = process.env.NACRE_SMOKE_ANTHROPIC_KEY
  ? { provider: 'anthropic', apiKey: process.env.NACRE_SMOKE_ANTHROPIC_KEY }
  : { provider: 'deepseek', apiKey: required('CAPTURE_LLM_API_KEY') };
const embedder = { provider: 'zhipu', apiKey: required('CAPTURE_EMBEDDER_API_KEY') };
console.log('LLM:', llm.provider, '| embedder: zhipu embedding-3');

function required(name) {
  const v = process.env[name];
  if (!v) {
    console.error(`${name} not set — source oracle/.env first`);
    process.exit(1);
  }
  return v;
}

const GROUP = 'smoke';
const turns = [
  ['turn-0', "Dana: Guess what — I finally adopted a greyhound! Her name is Comet.\nFelix: That's great! Where are you keeping her, your flat in Rotterdam?", '2026-05-04T10:00:00+00:00'],
  ['turn-1', "Dana: Yes, though Comet needs more space, so I'm moving to Utrecht next month.\nFelix: Makes sense. I'm still at Brightwater Analytics, by the way — got promoted to data lead.", '2026-05-11T09:30:00+00:00'],
  ['turn-2', "Dana: Congrats! Utrecht move is confirmed for June 1st, found a place near Wilhelminapark.\nFelix: Perfect for Comet. Bring her to the office sometime, Brightwater is dog-friendly now.", '2026-05-18T16:45:00+00:00'],
];

const dbPath = path.join(os.tmpdir(), `nacre-node-live-${process.pid}.db`);
const m = Memory.open(dbPath, 'node-live-smoke');
console.log('grit file:', dbPath);

for (const [name, content, at] of turns) {
  const out = await m.addEpisode(
    { name, content, sourceDescription: 'message', groupId: GROUP, validAt: at },
    llm,
    embedder,
  );
  console.log(
    `ingested ${name}: ${out.nodeIds.length} nodes, ${out.newEdgeIds.length} new edges, ` +
    `${out.merges.length} merges, ${out.invalidatedEdgeIds.length} invalidations`,
  );
}

const live = m.nodesInGroup(GROUP).filter((n) => !n.expiredAt);
console.log(`\ngraph: ${live.length} live nodes:`, live.map((n) => n.name));

for (const q of ['Where does Felix work?', 'What kind of dog is Comet?', 'Where is Dana moving?']) {
  const hits = await m.searchEdges(q, GROUP, 3, embedder);
  console.log(`\nquery: ${q}`);
  for (const h of hits) console.log(`  - ${h.fact} [episodes: ${h.episodes.length}]`);
  if (!hits.length) {
    console.error('  (no hits) — FAIL');
    process.exit(1);
  }
}

fs.unlinkSync(dbPath);
console.log('\nLIVE NODE SMOKE OK');
