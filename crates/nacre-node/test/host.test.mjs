// Offline test of the "host" provider: the addon makes NO network requests —
// chat/embed callbacks supplied from JS are the entire transport. The chat
// stub answers each schema (recognized by the schema block nacre appends to
// the prompt) with a minimal valid response; the embed stub returns
// deterministic vectors. Skips (loudly) when the addon has not been built.
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

const DIM = 8;

function makeStubs() {
  const chatRequests = [];
  const embedBatches = [];

  const llm = {
    provider: 'host',
    chat: async (request) => {
      chatRequests.push(request);
      const prompt = request.messages.map((m) => m.content).join('\n');
      // The schema block nacre appends names the required top-level keys;
      // answer whichever schema this round asked for. Extract one entity so
      // the pipeline exercises summaries and embeddings too.
      if (prompt.includes('"extracted_entities"')) {
        return JSON.stringify({
          extracted_entities: [{ name: 'Ada', entity_type_id: 0, episode_indices: [0] }],
        });
      }
      if (prompt.includes('"edges"')) return JSON.stringify({ edges: [] });
      if (prompt.includes('"entity_resolutions"')) {
        return JSON.stringify({ entity_resolutions: [] });
      }
      if (prompt.includes('"duplicate_facts"')) {
        return JSON.stringify({ duplicate_facts: [], contradicted_facts: [] });
      }
      if (prompt.includes('"summaries"')) {
        return JSON.stringify({ summaries: [{ name: 'Ada', summary: 'Ada is a person.' }] });
      }
      if (prompt.includes('"timestamps"')) return JSON.stringify({ timestamps: [] });
      if (prompt.includes('"description"')) return JSON.stringify({ description: 'd' });
      if (prompt.includes('"summary"')) return JSON.stringify({ summary: 's' });
      throw new Error(`host chat stub got an unrecognized prompt:\n${prompt.slice(0, 400)}`);
    },
  };

  const embedder = {
    provider: 'host',
    model: 'embedding-3',
    dim: DIM,
    embed: async (inputs) => {
      embedBatches.push(inputs);
      // Deterministic per-input vectors, deliberately longer than DIM to
      // prove the addon MRL-slices to the configured dimension.
      return inputs.map((_, i) =>
        Array.from({ length: DIM + 4 }, (_, j) => (i + 1) / (j + 1)),
      );
    },
  };

  return { llm, embedder, chatRequests, embedBatches };
}

test('host provider routes all I/O through JS callbacks', { skip: !Memory }, async () => {
  const dbPath = path.join(os.tmpdir(), `nacre-node-host-test-${process.pid}.db`);
  try { fs.unlinkSync(dbPath); } catch {}
  const m = Memory.open(dbPath, 'host-test');
  const { llm, embedder, chatRequests, embedBatches } = makeStubs();

  const outcome = await m.addEpisode(
    {
      name: 'ep-0',
      content: 'Ada wrote the first program.',
      sourceDescription: 'chat',
      groupId: 'host-group',
      validAt: '2026-07-14T00:00:00Z',
    },
    llm,
    embedder,
  );

  assert.equal(outcome.nodeIds.length, 1, 'one entity extracted via the chat callback');
  assert.ok(chatRequests.length >= 2, `chat callback drove the pipeline (${chatRequests.length} calls)`);
  assert.ok(embedBatches.length >= 1, 'embed callback produced vectors');
  assert.ok(
    chatRequests.every((r) => r.maxTokens > 0 && ['small', 'medium'].includes(r.modelSize)),
    'requests carry budget and tier',
  );

  const nodes = m.nodesInGroup('host-group');
  assert.ok(nodes.some((n) => n.name === 'Ada'));

  const hits = await m.searchEdges('who wrote programs?', 'host-group', 3, embedder);
  assert.ok(Array.isArray(hits), 'search runs through the host embedder');
  assert.ok(
    embedBatches.some((batch) => batch.includes('who wrote programs?')),
    'query embedding went through the callback',
  );

  // Callback rejections surface as provider errors, not crashes.
  await assert.rejects(
    () =>
      m.addEpisode(
        {
          name: 'ep-err',
          content: 'x',
          sourceDescription: 'chat',
          groupId: 'host-group',
          validAt: '2026-07-14T00:01:00Z',
        },
        { provider: 'host', chat: async () => { throw new Error('host transport down'); } },
        embedder,
      ),
    /host transport down|giving up/,
  );

  // Networked provider names are compiled out of the shipped addon.
  await assert.rejects(
    () =>
      m.searchEdges('q', 'host-group', 1, { provider: 'zhipu', apiKey: 'sk-x' }),
    /without networked providers/,
  );

  fs.unlinkSync(dbPath);
});
