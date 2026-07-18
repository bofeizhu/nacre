# viz — the graph-view data contract, proven

A minimal, self-contained memory-graph visualization. **Not a product** — a
proof that the read-path contract exposed by `nacre-node` is sufficient for
the Layer 3 graph view, and a reference the Layer 3 app can copy from.

## Files

| File | What |
|---|---|
| `dump-graph.mjs` | Replay-ingests the committed golden trace (`oracle/fixtures/trace1`) through the built addon — offline, no network — and dumps every row the view needs. |
| `viewer.html` | Self-contained viewer (no CDN, no build step, works from `file://`): force-directed graph, hover tooltips, an as-of **event-time** slider, and an audit toggle for merged-away drafts. Light/dark via `prefers-color-scheme`. |
| `graph.json` | The pure data contract — a committed sample of the dump. |
| `graph.data.js` | The same object as `window.NACRE_GRAPH = …`, so `viewer.html` loads it via `<script src>` without a server (browsers block `fetch` on `file://`). |

## Run it

```sh
cd crates/nacre-node
npm install && npm run build   # build the addon once
node examples/viz/dump-graph.mjs
open examples/viz/viewer.html  # or just double-click it
```

The committed `graph.json` / `graph.data.js` are deterministic replays of
trace1, so the viewer renders straight from a checkout too. (Node/edge *ids*
are freshly assigned per dump run; everything else is byte-stable.)

## The data contract

Everything below comes from five calls — this is the entire surface the
graph view needs:

```js
const nodes    = m.nodesInGroup(groupId);    // full bi-temporal rows
const edges    = m.edgesInGroup(groupId);
const episodes = m.episodesInGroup(groupId);
const prov     = m.mentionsOf(edgeId);       // edge → provenance episode ids
const sub      = m.traverse(seedIds, opts);  // bounded neighborhoods (not
                                             // used here — the sample fits
                                             // on one screen)
```

How the viewer reads the bi-temporal fields:

- **Live view** (default): nodes with `expiredAt == null`, edges with
  `expiredAt == null`. `expiredAt` is *belief* time — a retracted row, e.g.
  a draft that merged away.
- **As-of slider** (*event* time `t`): an edge is *not yet stated* when
  `validAt > t`, *invalidated* (dashed) when `invalidAt <= t`, *current*
  otherwise. This is the stub for the app's full `traverse({asOf, asAt})`
  time travel.
- **Audit toggle**: shows expired nodes as ghosts with a dotted link to
  `mergedInto` — the dedup story is in the data, not just the deltas.
- **Provenance**: edge tooltips list the episodes (`source`, `occurredAt`)
  each fact traces to, via `mentionsOf`.
