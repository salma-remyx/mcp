import { Tool } from '../../core/tool';

/**
 * Query-based tool gating — selects the subset of tools relevant to a user query so that only
 * their input schemas are materialized for the model on a given turn.
 *
 * Eagerly injecting the full JSON schema of every registered tool each turn imposes a per-turn
 * token overhead (the "MCP/Tools Tax"). Gating reduces that payload by exposing only the tools
 * the current query is likely to need, while keeping an explicit escape hatch (e.g. the
 * manage-tools tool) available so the agent can still request more.
 *
 * Adapted from "Tool Attention Is All You Need: Dynamic Tool Gating and Lazy Schema Loading for
 * Eliminating the MCP/Tools Tax in Scalable Agentic Workflows" (arXiv:2604.21816). The paper's
 * core mechanism — dynamic per-query gating of the tool set — is preserved at full fidelity. Its
 * learned attention selector is substituted with the parameter-free lexical relevance proxy below
 * (term overlap between the query and each tool's name + description), which needs no training,
 * weights, or model calls. The paper's separate benchmark/evaluation framework is intentionally
 * out of scope here.
 */

/** Common English tokens that carry little retrieval signal. */
const STOPWORDS = new Set([
  'a',
  'an',
  'and',
  'are',
  'as',
  'at',
  'be',
  'by',
  'for',
  'from',
  'has',
  'in',
  'is',
  'it',
  'its',
  'of',
  'on',
  'or',
  'that',
  'the',
  'to',
  'use',
  'using',
  'with',
  'i',
  'me',
  'my',
  'we',
  'you',
  'your',
]);

/** Tokens found in a tool's name rank this much higher than tokens found only in its description. */
const NAME_WEIGHT = 3;

export interface ToolGatingOptions {
  /** Maximum number of tools to keep after gating. When omitted, every positively-matching tool is kept. */
  maxTools?: number;
  /** Minimum relevance score in [0, 1] required for a tool to be kept. Defaults to 0 (keep any match). */
  minScore?: number;
  /** Tool names that are always kept regardless of their score (e.g. the manage-tools escape hatch). */
  alwaysInclude?: string[];
}

/**
 * Split free text into normalized retrieval tokens: lower-cased alphanumeric runs, dropping
 * stop-words and tokens shorter than two characters.
 */
export function tokenize(text: string): string[] {
  if (!text) {
    return [];
  }
  return text
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((token) => token.length >= 2 && !STOPWORDS.has(token));
}

/**
 * Score how relevant a tool is to a query on [0, 1]. A query token covered by the tool's name
 * contributes `NAME_WEIGHT`; a token covered only by its description contributes 1. The score is
 * the weighted coverage normalized by the number of query tokens, so a tool whose name covers the
 * whole query scores 1. Returns 0 when there is no overlap or the query has no tokens.
 */
export function scoreToolRelevance(tool: Tool<any, any>, query: string): number {
  const queryTokens = tokenize(query);
  if (queryTokens.length === 0) {
    return 0;
  }

  const nameTokens = new Set(tokenize(tool.name));
  const descriptionTokens = new Set(tokenize(tool.getDescription()));

  let nameHits = 0;
  let descriptionHits = 0;
  for (const token of queryTokens) {
    if (nameTokens.has(token)) {
      nameHits++;
    } else if (descriptionTokens.has(token)) {
      descriptionHits++;
    }
  }

  if (nameHits === 0 && descriptionHits === 0) {
    return 0;
  }

  const score = (nameHits * NAME_WEIGHT + descriptionHits) / (queryTokens.length * NAME_WEIGHT);
  return Math.min(1, score);
}

/**
 * Return the subset of `tools` relevant to `query`, ranked by relevance.
 *
 * Behavior:
 *  - No query (empty/whitespace) returns all tools unchanged, preserving the default eager
 *    behavior — gating is strictly opt-in.
 *  - Tools listed in `alwaysInclude` are always kept and ranked first, so the gating escape hatch
 *    stays reachable even when it does not lexically match the query.
 *  - If nothing matches and nothing is force-included, all tools are returned rather than leaving
 *    the agent with no tools.
 */
export function gateTools(
  tools: Tool<any, any>[],
  query: string | undefined,
  options?: ToolGatingOptions,
): Tool<any, any>[] {
  if (!query || !query.trim()) {
    return tools;
  }

  const maxTools = options?.maxTools;
  const minScore = options?.minScore ?? 0;
  const alwaysInclude = new Set(options?.alwaysInclude ?? []);

  const scored = tools.map((tool, index) => ({
    tool,
    index,
    score: alwaysInclude.has(tool.name) ? Infinity : scoreToolRelevance(tool, query),
  }));

  const kept = scored.filter((entry) => alwaysInclude.has(entry.tool.name) || entry.score > minScore);
  if (kept.length === 0) {
    return tools;
  }

  kept.sort((a, b) => {
    if (b.score !== a.score) {
      return b.score - a.score;
    }
    return a.index - b.index; // stable ordering for ties
  });

  const selected = typeof maxTools === 'number' && maxTools > 0 ? kept.slice(0, maxTools) : kept;
  return selected.map((entry) => entry.tool);
}
