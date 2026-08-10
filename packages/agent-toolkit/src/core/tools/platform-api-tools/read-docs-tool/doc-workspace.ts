import { existsSync, mkdirSync, readFileSync, readdirSync, rmSync, statSync, unlinkSync, writeFileSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';

/**
 * Persistent document workspace for the read_docs tool.
 *
 * Adapted from "Fetch-then-Explore: Decoupling Selection from Extraction over a
 * Persistent Workspace for Search Agents" (arXiv:2608.02097). The paper's core
 * mechanism — record a fetched page to a filesystem workspace at selection time
 * so evidence can be pulled from it on demand later, instead of re-fetching and
 * re-rendering the whole page into context — is applied to monday.com documents:
 * a `read_docs` "content" fetch (selection) records the doc's markdown + blocks
 * here, and a later `read_docs` "explore_cache" call (extraction) pulls just the
 * relevant snippet or block without another API round-trip. Selection becomes
 * nearly free, and a page the agent left can be revisited many turns later.
 *
 * Intentionally NOT ported (Mode 2 substitutions): the paper's ReAct agent loop,
 * its three agent backbones, and the BrowseComp/WideSearch benchmark harness.
 * This repo ships tools *to* agents rather than running an agent, so only the
 * persistent-workspace primitive maps; benchmark evaluation belongs downstream.
 */

const DEFAULT_BASE_DIR = join(tmpdir(), 'monday-mcp-doc-workspace');

// Best-effort cap so a long-lived process can't grow the workspace unbounded.
// Eviction is oldest-file-first (by mtime), approximating the paper's notion of
// evidence accumulating across a trajectory while still bounding disk use.
const MAX_CACHED_DOCS = 50;

export interface CachedDoc {
  id: string;
  name?: string | null;
  object_id?: string | null;
  blocks_as_markdown: string;
  blocks?: Array<{ id: string; type: string; content: unknown }>;
  recorded_at?: string;
}

export interface ExtractOptions {
  /** Case-insensitive substring to search for within the cached markdown. */
  query?: string;
  /** Specific block ids to pull verbatim from the cached doc. */
  block_ids?: string[];
  /** Maximum number of matches to return (default: 25). */
  limit?: number;
}

export interface ExtractMatch {
  block_id?: string;
  type?: string;
  markdown: string;
}

export interface ExtractResult {
  doc_id: string;
  cached: true;
  query?: string;
  matches: ExtractMatch[];
  total_matches: number;
}

function baseDir(): string {
  return process.env.MONDAY_DOC_WORKSPACE_DIR || DEFAULT_BASE_DIR;
}

// Keep the doc id from escaping the workspace directory.
function docPath(docId: string): string {
  const safe = docId.replace(/[^a-zA-Z0-9_-]/g, '_');
  return join(baseDir(), `${safe}.json`);
}

function evictIfNeeded(): void {
  let files: Array<{ name: string; mtime: number }>;
  try {
    files = readdirSync(baseDir())
      .filter((f) => f.endsWith('.json'))
      .map((name) => ({ name, mtime: statSync(join(baseDir(), name)).mtimeMs }));
  } catch {
    return;
  }
  const excess = files.length - MAX_CACHED_DOCS;
  if (excess <= 0) return;
  files
    .sort((a, b) => a.mtime - b.mtime)
    .slice(0, excess)
    .forEach((f) => {
      try {
        unlinkSync(join(baseDir(), f.name));
      } catch {
        // ignore — best-effort eviction
      }
    });
}

/** Record fetched documents to the persistent workspace (the "selection" half). */
export function recordDocs(docs: CachedDoc[]): void {
  for (const doc of docs) {
    try {
      mkdirSync(baseDir(), { recursive: true });
      const payload: CachedDoc = { ...doc, recorded_at: new Date().toISOString() };
      writeFileSync(docPath(doc.id), JSON.stringify(payload), 'utf8');
    } catch {
      // The cache is best-effort: a write failure must never break a read_docs call.
    }
  }
  evictIfNeeded();
}

/** Load a cached document by id, or null if it was never recorded. */
export function getCachedDoc(docId: string): CachedDoc | null {
  try {
    const filePath = docPath(docId);
    if (!existsSync(filePath)) return null;
    return JSON.parse(readFileSync(filePath, 'utf8')) as CachedDoc;
  } catch {
    return null;
  }
}

/** List the ids (and names, when available) of documents in the workspace. */
export function listCachedDocs(): Array<{ id: string; name?: string | null }> {
  const out: Array<{ id: string; name?: string | null }> = [];
  try {
    const dir = baseDir();
    if (!existsSync(dir)) return [];
    for (const f of readdirSync(dir)) {
      if (!f.endsWith('.json')) continue;
      try {
        const doc = JSON.parse(readFileSync(join(dir, f), 'utf8')) as CachedDoc;
        out.push({ id: doc.id, name: doc.name });
      } catch {
        // skip unreadable cache entries
      }
    }
  } catch {
    // ignore — return whatever was collected
  }
  return out;
}

/** Remove every document from the workspace. */
export function clearWorkspace(): void {
  try {
    rmSync(baseDir(), { recursive: true, force: true });
  } catch {
    // ignore
  }
}

/**
 * Render a single cached block to a readable string. monday doc block content is
 * either a serialized-JSON string or a parsed object; both shapes are handled.
 */
function blockContentToText(content: unknown): string {
  let parsed: Record<string, unknown> | null = null;
  if (typeof content === 'string') {
    try {
      parsed = JSON.parse(content) as Record<string, unknown>;
    } catch {
      return content;
    }
  } else if (content && typeof content === 'object') {
    parsed = content as Record<string, unknown>;
  }
  if (!parsed) return '';

  if (typeof parsed.text === 'string') return parsed.text;
  const delta = parsed.deltaFormat;
  if (Array.isArray(delta)) {
    return delta
      .map((op) =>
        typeof (op as Record<string, unknown>)?.insert === 'string' ? (op as { insert: string }).insert : '',
      )
      .join('');
  }
  try {
    return JSON.stringify(parsed);
  } catch {
    return '';
  }
}

/**
 * Pull targeted evidence from a cached document on demand (the "explore" half).
 * Prefers explicit block_ids, then a keyword query over the markdown, and falls
 * back to returning the whole cached doc when no filter is given.
 */
export function extractFromCache(docId: string, opts: ExtractOptions = {}): { found: false } | ExtractResult {
  const doc = getCachedDoc(docId);
  if (!doc) return { found: false };

  const limit = opts.limit && opts.limit > 0 ? opts.limit : 25;

  if (opts.block_ids && opts.block_ids.length > 0) {
    const wanted = new Set(opts.block_ids);
    const matches = (doc.blocks ?? [])
      .filter((b) => wanted.has(b.id))
      .slice(0, limit)
      .map((b) => ({ block_id: b.id, type: b.type, markdown: blockContentToText(b.content) }));
    return { doc_id: docId, cached: true, matches, total_matches: matches.length };
  }

  if (opts.query && opts.query.trim().length > 0) {
    const needle = opts.query.toLowerCase();
    const matches = doc.blocks_as_markdown
      .split(/\n+/)
      .map((line) => line.trim())
      .filter((line) => line.length > 0 && line.toLowerCase().includes(needle))
      .slice(0, limit)
      .map((markdown) => ({ markdown }));
    return { doc_id: docId, cached: true, query: opts.query, matches, total_matches: matches.length };
  }

  return { doc_id: docId, cached: true, matches: [{ markdown: doc.blocks_as_markdown }], total_matches: 1 };
}

/**
 * Explore a cached document and return a tool-output-shaped result: either the
 * extracted matches or a human-readable message when the doc isn't cached.
 */
export function exploreCachedDoc(
  docId: string,
  opts: ExtractOptions = {},
): { content: string } | { content: ExtractResult } {
  const result = extractFromCache(docId, opts);
  if ('found' in result) {
    const cached = listCachedDocs();
    const hint =
      cached.length > 0
        ? ` Cached doc ids: ${cached.map((d) => d.id).join(', ')}.`
        : ' No documents are cached yet — call read_docs in "content" mode first.';
    return { content: `Document ${docId} is not in the workspace cache.${hint}` };
  }
  return { content: result };
}
