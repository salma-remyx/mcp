/**
 * Per-turn tool gating by intent–schema overlap, plus a two-phase lazy schema
 * loader that splits the tool catalog into a cheap "summary pool" (sent for
 * every tool every turn) and an on-demand "detail pool" (full schemas for the
 * gated subset only). The goal is to shrink the per-turn "MCP/Tools Tax" — the
 * schema payload injected into the model context — without dropping tools the
 * agent may actually need.
 *
 * Adapted from "Tool Attention Is All You Need: Dynamic Tool Gating and Lazy
 * Schema Loading for Eliminating the MCP/Tools Tax in Scalable Agentic
 * Workflows" (arXiv:2604.21816).
 *
 * Mode 2 (adapted port): the paper's core mechanism — Intent–Schema overlap
 * scoring, state-aware gating, and the two-phase lazy schema loader — is kept
 * at full fidelity. Auxiliary components are target-native substitutions:
 *   - The paper's intent/schema relevance estimator is replaced with a
 *     parameter-free lexical token-overlap proxy (no model or embeddings).
 *   - The paper's benchmark harness for the "MCP/Tools Tax" is cut; a small
 *     chars/4 token estimate is exposed for the same signal instead.
 *   - Gating is applied through the repo's existing DynamicToolManager
 *     enable/disable primitive rather than a bespoke middleware transport.
 */

/** Minimal tool metadata needed to score a tool against a turn intent. */
export interface ToolGateMeta {
  name: string;
  description: string;
  /** Parameter names — the "schema vocabulary" used for overlap scoring. */
  paramNames?: readonly string[];
}

export interface GateOptions {
  /** Maximum tools kept in the detail pool (the enabled set). Default 8. */
  topK?: number;
  /** Tool names never gated off, regardless of overlap score. */
  alwaysOn?: readonly string[];
  /** Minimum normalized overlap [0,1] to keep a tool in the detail pool. Default 0. */
  minScore?: number;
}

/** A short, cheap summary entry sent for every tool (phase 1). */
export interface ToolSummary {
  name: string;
  summary: string;
}

export interface TokenTax {
  /** Estimated full-schema tokens injected if every tool is sent. */
  allTools: number;
  /** Estimated full-schema tokens injected for the gated detail pool only. */
  gated: number;
  /** Tokens avoided this turn by lazy-loading. */
  saved: number;
}

export interface GateResult {
  /** Tools whose full schemas are loaded this turn (the gated detail pool). */
  detailPool: string[];
  /** One-line summaries for ALL tools, sent cheaply every turn. */
  summaryPool: ToolSummary[];
  /** Per-tool overlap scores, descending. */
  scores: Array<{ name: string; score: number }>;
  /** Estimated "MCP/Tools Tax" for all vs. gated pools. */
  tokenTax: TokenTax;
}

const DEFAULT_TOP_K = 8;
const NAME_BOOST = 0.25;
const SUMMARY_MAX_CHARS = 90;

const STOPWORDS = new Set([
  'a',
  'an',
  'the',
  'and',
  'or',
  'of',
  'to',
  'in',
  'on',
  'for',
  'with',
  'by',
  'is',
  'are',
  'be',
  'this',
  'that',
  'it',
  'as',
  'at',
  'from',
  'into',
  'using',
  'use',
  'tool',
  'tools',
  'monday',
  'api',
  'get',
  'set',
  'new',
  'list',
  'all',
]);

/** Split snake_case / camelCase / free text into lowercase content tokens. */
function tokenize(text: string): string[] {
  return text
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((token) => token.length > 1 && !STOPWORDS.has(token));
}

function buildVocabulary(meta: ToolGateMeta): { vocab: Set<string>; names: Set<string> } {
  const nameTokens = tokenize(meta.name);
  const descriptionTokens = tokenize(meta.description);
  const paramTokens = (meta.paramNames ?? []).flatMap(tokenize);
  return {
    vocab: new Set([...nameTokens, ...descriptionTokens, ...paramTokens]),
    names: new Set(nameTokens),
  };
}

function scoreTokens(intentTokens: readonly string[], meta: ToolGateMeta): number {
  const intent = new Set(intentTokens);
  if (intent.size === 0) return 0;
  const { vocab, names } = buildVocabulary(meta);
  let matched = 0;
  let nameMatched = 0;
  intent.forEach((token) => {
    if (vocab.has(token)) matched += 1;
    if (names.has(token)) nameMatched += 1;
  });
  const coverage = matched / intent.size;
  const nameCoverage = nameMatched / intent.size;
  return Math.min(1, coverage + NAME_BOOST * nameCoverage);
}

/**
 * Intent–Schema overlap score in [0,1]: the fraction of the intent's content
 * tokens present in the tool's schema vocabulary (name + description + params),
 * with a small boost when the overlap lands on the tool's own name tokens.
 */
export function scoreToolIntent(intent: string, meta: ToolGateMeta): number {
  return scoreTokens(tokenize(intent), meta);
}

/** Compress a description to a single short line for the summary pool. */
export function summarizeDescription(description: string, maxChars = SUMMARY_MAX_CHARS): string {
  const firstSentence = description.split(/(?<=[.!?])\s+/)[0] ?? description;
  const trimmed = firstSentence.trim().replace(/\s+/g, ' ');
  if (trimmed.length <= maxChars) return trimmed;
  const cut = trimmed.slice(0, maxChars - 1);
  const lastSpace = cut.lastIndexOf(' ');
  return cut.slice(0, lastSpace > 0 ? lastSpace : cut.length).trimEnd() + '…';
}

/**
 * Two-phase lazy gate. Phase 1 emits a summary entry for every tool (cheap,
 * always sent). Phase 2 selects the detail pool — the top-K tools by overlap
 * score (≥ minScore), unioned with always-on tools — whose full schemas are
 * loaded on demand this turn.
 */
export function gate(metas: readonly ToolGateMeta[], intent: string, options: GateOptions = {}): GateResult {
  const topK = options.topK ?? DEFAULT_TOP_K;
  const minScore = options.minScore ?? 0;
  const alwaysOn = new Set(options.alwaysOn ?? []);
  const intentTokens = tokenize(intent);

  const scored = metas
    .map((meta) => ({ name: meta.name, score: scoreTokens(intentTokens, meta) }))
    .sort((a, b) => b.score - a.score || (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));

  const detailPool: string[] = [];
  for (const { name, score } of scored) {
    if (detailPool.length >= topK && !alwaysOn.has(name)) continue;
    if (alwaysOn.has(name) || score >= minScore) {
      detailPool.push(name);
    }
  }

  const summaryPool: ToolSummary[] = metas.map((meta) => ({
    name: meta.name,
    summary: summarizeDescription(meta.description),
  }));

  return { detailPool, summaryPool, scores: scored, tokenTax: estimateTokenTax(metas, detailPool) };
}

/** Select only the gated detail-pool tool names for a turn intent. */
export function selectToolsByIntent(
  metas: readonly ToolGateMeta[],
  intent: string,
  options: GateOptions = {},
): string[] {
  return gate(metas, intent, options).detailPool;
}

/**
 * Estimate the per-turn "MCP/Tools Tax" — full-schema tokens injected — for the
 * whole catalog vs. the gated detail pool. Uses chars/4 as a parameter-free
 * proxy for token count; precise per-schema measurement belongs downstream.
 */
export function estimateTokenTax(metas: readonly ToolGateMeta[], detailPool: readonly string[]): TokenTax {
  const perTool = new Map<string, number>();
  metas.forEach((meta) => {
    const paramChars = (meta.paramNames ?? []).join(' ').length;
    const chars = meta.name.length + meta.description.length + paramChars;
    perTool.set(meta.name, Math.ceil(chars / 4));
  });
  const detail = new Set(detailPool);
  const allTools = [...perTool.values()].reduce((sum, tokens) => sum + tokens, 0);
  const gated = [...perTool.entries()]
    .filter(([name]) => detail.has(name))
    .reduce((sum, [, tokens]) => sum + tokens, 0);
  return { allTools, gated, saved: Math.max(0, allTools - gated) };
}
