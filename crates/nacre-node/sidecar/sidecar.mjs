// nacre sidecar — ndjson-over-stdio server over the built addon.
//
// The Hermes memory provider (integrations/hermes/) spawns this as a child
// process and drives it line-by-line: one JSON request per stdin line, one
// JSON response per stdout line. stdio (not a port) so the process dies
// with its parent, needs no port allocation, and is trivially sandboxed.
//
// Requests are processed STRICTLY IN ORDER (one at a time): the pipeline
// mutates one graph, and episode order is meaningful.
//
// Request:  {"id": <any>, "method": "<name>", "params": {...}}
// Response: {"id": <same>, "result": {...}}   on success
//           {"id": <same>, "error": "<message>"}   on failure (fail-loud;
//           the caller decides whether to retry, skip, or trip a breaker)
//
// Methods: see README.md next to this file for the full shapes.
//
// Provider credentials come from the ENVIRONMENT (never argv, never a
// request): NACRE_LLM_* and NACRE_EMBEDDER_* — see buildConfigs().
import path from 'node:path';
import readline from 'node:readline';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const { Memory, version } = await import(path.join(here, '..', 'index.js'));

// LlmConfig / EmbedderConfig for Memory.addEpisode / searchEdges, from env.
// Replay mode (offline tests): NACRE_LLM_PROVIDER=replay with
// NACRE_LLM_RECORDINGS / NACRE_EMBEDDER_RECORDINGS paths.
function buildConfigs(env) {
  const llmProvider = env.NACRE_LLM_PROVIDER || 'deepseek';
  const llm =
    llmProvider === 'replay'
      ? { provider: 'replay', recordingsPath: required(env, 'NACRE_LLM_RECORDINGS') }
      : {
          provider: llmProvider,
          apiKey: required(env, 'NACRE_LLM_API_KEY'),
          ...(env.NACRE_LLM_BASE_URL && { baseUrl: env.NACRE_LLM_BASE_URL }),
          ...(env.NACRE_LLM_MODEL && { mediumModel: env.NACRE_LLM_MODEL }),
          ...(env.NACRE_LLM_SMALL_MODEL && { smallModel: env.NACRE_LLM_SMALL_MODEL }),
        };
  const embProvider = env.NACRE_EMBEDDER_PROVIDER || 'zhipu';
  const embedder =
    embProvider === 'replay'
      ? { provider: 'replay', recordingsPath: required(env, 'NACRE_EMBEDDER_RECORDINGS') }
      : {
          provider: embProvider,
          apiKey: required(env, 'NACRE_EMBEDDER_API_KEY'),
          ...(env.NACRE_EMBEDDER_BASE_URL && { baseUrl: env.NACRE_EMBEDDER_BASE_URL }),
          ...(env.NACRE_EMBEDDER_MODEL && { model: env.NACRE_EMBEDDER_MODEL }),
          ...(env.NACRE_EMBEDDER_DIM && { dim: Number(env.NACRE_EMBEDDER_DIM) }),
        };
  return { llm, embedder };
}

function required(env, name) {
  const v = env[name];
  if (!v) throw new Error(`missing required environment variable ${name}`);
  return v;
}

// --- state (set by init) ---
let memory = null;
let configs = null;
let defaultGroupId = null;

const handlers = {
  // {dbPath, deviceId, groupId} → {version, dbPath, groupId}
  init(params) {
    for (const key of ['dbPath', 'deviceId', 'groupId']) {
      if (!params?.[key]) throw new Error(`init: missing ${key}`);
    }
    configs = buildConfigs(process.env); // validate env up front, fail loud
    memory = Memory.open(params.dbPath, params.deviceId);
    defaultGroupId = params.groupId;
    return { version: version(), dbPath: params.dbPath, groupId: defaultGroupId };
  },

  // {content, sourceDescription, name?, source?, groupId?, validAt?, id?,
  //  metadata?} → AddEpisodeResult {episodeId, nodeIds, newEdgeIds, merges,
  //  invalidatedEdgeIds}
  //
  // `id` is the caller's idempotency key: pass the same one on every retry
  // of an episode and a duplicate ingest is refused (EPISODE_EXISTS) rather
  // than written twice. `metadata` is the caller's own payload, stored
  // verbatim and never shown to a model.
  async addEpisode(params) {
    requireInit();
    if (!params?.content) throw new Error('addEpisode: missing content');
    if (!params?.sourceDescription) throw new Error('addEpisode: missing sourceDescription');
    return await memory.addEpisode(
      {
        id: params.id,
        name: params.name ?? '',
        content: params.content,
        source: params.source,
        sourceDescription: params.sourceDescription,
        groupId: params.groupId ?? defaultGroupId,
        validAt: params.validAt,
        metadata: params.metadata,
      },
      configs.llm,
      configs.embedder,
    );
  },

  // {query, groupId?, limit?} → {hits: [{id, sourceId, targetId, name, fact,
  //   validAt, invalidAt, episodes}]}
  async searchEdges(params) {
    requireInit();
    if (!params?.query) throw new Error('searchEdges: missing query');
    const hits = await memory.searchEdges(
      params.query,
      params.groupId ?? defaultGroupId,
      params.limit ?? 10,
      configs.embedder,
    );
    return { hits };
  },

  // {id} → {episode: EpisodeRow | null} — the point lookup behind a
  // provenance id from searchEdges, and how a caller reads its own
  // `metadata` back. An id the graph does not hold is `null`, not an error.
  episode(params) {
    requireInit();
    if (!params?.id) throw new Error('episode: missing id');
    return { episode: memory.episode(params.id) ?? null };
  },

  // {id} → DeleteEpisodeResult {episodeId, purgedNodeIds, purgedEdgeIds,
  //   keptNodeIds, keptEdgeIds}
  //
  // Forgets one episode and the graph that exists only because of it.
  // An unknown id (or a second delete) is an error whose message leads
  // with EPISODE_NOT_FOUND; nothing is written in that case. The id is
  // spent afterwards — re-ingesting the same memory needs a new one.
  deleteEpisode(params) {
    requireInit();
    if (!params?.id) throw new Error('deleteEpisode: missing id');
    return memory.deleteEpisode(params.id);
  },

  // {groupId?} → DeleteGroupResult {purgedNodeIds, purgedEdgeIds,
  //   purgedEpisodeIds}
  //
  // Forgets a whole namespace (defaults to init's groupId). An unknown or
  // empty group is not an error: zero counts, nothing written.
  deleteGroup(params) {
    requireInit();
    return memory.deleteGroup(params?.groupId ?? defaultGroupId);
  },

  // {} → {version, groupId, episodes, liveNodes, edges} (uninitialized: {version})
  status() {
    if (!memory) return { version: version(), initialized: false };
    const nodes = memory.nodesInGroup(defaultGroupId);
    return {
      version: version(),
      initialized: true,
      groupId: defaultGroupId,
      episodes: memory.episodesInGroup(defaultGroupId).length,
      liveNodes: nodes.filter((n) => !n.expiredAt).length,
      edges: memory.edgesInGroup(defaultGroupId).length,
    };
  },

  // {} → {ok: true}; the process exits after the response is flushed.
  shutdown() {
    setImmediate(() => process.exit(0));
    return { ok: true };
  },
};

function requireInit() {
  if (!memory) throw new Error('not initialized — call init first');
}

const respond = (obj) => process.stdout.write(JSON.stringify(obj) + '\n');

// Strictly sequential: each request awaits the previous one.
let chain = Promise.resolve();
const rl = readline.createInterface({ input: process.stdin, terminal: false });
rl.on('line', (line) => {
  if (!line.trim()) return;
  chain = chain.then(async () => {
    let id = null;
    try {
      const req = JSON.parse(line);
      id = req.id ?? null;
      const handler = handlers[req.method];
      if (!handler) throw new Error(`unknown method ${JSON.stringify(req.method)}`);
      respond({ id, result: await handler(req.params) });
    } catch (e) {
      respond({ id, error: String(e?.message ?? e) });
    }
  });
});
rl.on('close', () => {
  // Parent closed stdin (or died): exit once in-flight work drains.
  chain.finally(() => process.exit(0));
});
