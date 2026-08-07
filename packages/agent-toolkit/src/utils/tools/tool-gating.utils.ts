/**
 * Query-driven tool gating — a parameter-free relevance gate that decides
 * which of an MCP server's tools should stay *enabled* for a given turn.
 *
 * Background / attribution:
 *   Adapted from "Tool Attention Is All You Need: Dynamic Tool Gating and
 *   Lazy Schema Loading for Eliminating the MCP/Tools Tax in Scalable
 *   Agentic Workflows" (arXiv:2604.21816). That paper observes that MCP
 *   servers eagerly inject every tool's schema on every turn — a per-turn
 *   "MCP/Tools Tax" practitioner reports place around 10k–60k tokens — and
 *   proposes a learned "ISO predictive gate" plus two-phase lazy schema
 *   loading to keep only the relevant tools in context.
 *
 * This is an ADAPTED PORT (Mode 2): the paper's learned gate is replaced by
 * a parameter-free query/description relevance proxy (tokenized term
 * overlap with a small name-boost), and the paper's bespoke two-phase
 * schema transport is replaced by the repo's *already-present* dynamic
 * enable/disable plumbing (DynamicToolManager + listChanged). The module
 * only computes a gating PLAN; mutating the toolkit's enabled set is the
 * caller's job (see MondayAgentToolkit.gateToolsForQuery). The token-budget
 * option is a parameter-free stand-in for the paper's lazy-schema cap: it
 * greedily admits tools by relevance until the estimated schema-payload
 * budget is reached, deferring the rest.
 *
 * The module is deliberately dependency-free and pure so it can be unit
 * tested without an MCP server or API client.
 */

/** A tool reduced to the surface this gate reasons over. */
export interface GateableTool {
  name: string;
  description: string;
}

/** Knobs for planToolGating. All optional; defaults keep relevant tools. */
export interface ToolGatingOptions {
  /** Hard cap on the number of tools to enable (top-K by relevance). */
  maxTools?: number;
  /**
   * Estimated schema-payload token budget. Tools are admitted by descending
   * relevance until the next one would exceed the budget; the rest defer.
   * Acts as the lazy-schema cap from the paper.
   */
  tokenBudget?: number;
  /** Minimum relevance score in [0,1] for a tool to be considered relevant. */
  minRelevance?: number;
}

/** The result of gating: who stays enabled, who defers, and the tax impact. */
export interface ToolGatingResult {
  /** Tool names to keep enabled this turn (most relevant first). */
  enabled: string[];
  /** Tool names to disable this turn (deferred out of the payload). */
  disabled: string[];
  /** Per-tool relevance scores in [0,1]. */
  scores: Record<string, number>;
  /** Estimated tokens of the enabled tools' schema payload. */
  estimatedEnabledTokens: number;
  /** Estimated tokens removed from the payload by deferring tools. */
  estimatedSavedTokens: number;
}

/** Common English stopwords ignored during tokenization. */
const STOP_WORDS = new Set([
  'a',
  'an',
  'the',
  'and',
  'or',
  'but',
  'if',
  'then',
  'else',
  'for',
  'of',
  'to',
  'in',
  'on',
  'at',
  'by',
  'with',
  'from',
  'as',
  'is',
  'are',
  'be',
  'been',
  'being',
  'this',
  'that',
  'these',
  'those',
  'it',
  'its',
  'i',
  'we',
  'you',
  'they',
  'my',
  'our',
  'your',
  'their',
  'me',
  'us',
  'them',
  'do',
  'does',
  'did',
  'can',
  'could',
  'should',
  'would',
  'will',
  'shall',
  'may',
  'might',
  'must',
  'have',
  'has',
  'had',
  'new',
  'some',
  'any',
  'all',
  'about',
  'into',
  'want',
  'needs',
  'need',
  'please',
]);

/**
 * Tokenize free text into normalized, stopword-filtered terms. Splits on
 * non-alphanumeric characters (so snake_case tool names yield their parts)
 * and drops stopwords + single characters.
 */
export const tokenize = (text: string): string[] => {
  if (!text) {
    return [];
  }
  return text
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((token) => token.length > 1 && !STOP_WORDS.has(token));
};

/**
 * Rough estimate of LLM tokens for a string. Uses the standard ~4
 * characters-per-token heuristic — a deliberately coarse, dependency-free
 * proxy adequate for ranking and budgeting, not for exact billing.
 */
export const estimateTokenCount = (text: string): number => {
  if (!text) {
    return 0;
  }
  return Math.max(1, Math.ceil(text.length / 4));
};

/**
 * Estimated per-turn schema-payload tokens for a tool. The description
 * dominates these tools' payload, so we estimate from name + description
 * rather than materializing the full JSON schema.
 */
export const estimateToolTokens = (tool: GateableTool): number => {
  return estimateTokenCount(`${tool.name} ${tool.description}`);
};

/**
 * Relevance of a single tool to a query, in [0,1]. Parameter-free: the
 * fraction of distinct query terms that appear in the tool's name or
 * description, with terms found in the tool *name* counted twice (names are
 * strong signals). Returns 0 when the query has no scorable terms.
 *
 * @param queryTerms Pre-tokenized query terms (pass the query string to have
 *   it tokenized here, or reuse already-tokenized terms for batch calls).
 */
export const scoreToolRelevance = (query: string | string[], tool: GateableTool): number => {
  const queryTerms = Array.isArray(query) ? query : tokenize(query);
  const distinctQuery = Array.from(new Set(queryTerms));
  if (distinctQuery.length === 0) {
    return 0;
  }

  const nameTerms = new Set(tokenize(tool.name));
  const descriptionTerms = new Set(tokenize(tool.description));

  let hits = 0;
  for (const term of distinctQuery) {
    if (nameTerms.has(term)) {
      hits += 2;
    } else if (descriptionTerms.has(term)) {
      hits += 1;
    }
  }

  // Normalize against the maximum possible weight (2 per distinct term).
  return Math.min(1, hits / (distinctQuery.length * 2));
};

/**
 * Compute a gating plan: which tools to enable vs. defer for `query`, and
 * the estimated effect on the per-turn schema payload. Pure — does not
 * touch any toolkit state.
 *
 * Selection order:
 *   1. Score every tool; keep those at/above `minRelevance` (default 0).
 *   2. Rank remaining candidates by descending relevance.
 *   3. If `tokenBudget` is set, greedily admit by rank until the next tool
 *      would exceed the budget (the lazy-schema cap).
 *      Else if `maxTools` is set, take the top-K.
 *      Else admit every relevant candidate.
 *   4. Safety net: if nothing is relevant, keep the single highest-scoring
 *      tool enabled so the agent is never left with zero tools.
 */
export const planToolGating = (query: string, tools: GateableTool[], options?: ToolGatingOptions): ToolGatingResult => {
  const minRelevance = options?.minRelevance ?? 0;
  const queryTerms = tokenize(query);

  const scored = tools.map((tool) => ({
    name: tool.name,
    tool,
    score: scoreToolRelevance(queryTerms, tool),
    tokens: estimateToolTokens(tool),
  }));

  const scores: Record<string, number> = {};
  const totalTokens = scored.reduce((sum, entry) => sum + entry.tokens, 0);
  for (const entry of scored) {
    scores[entry.name] = entry.score;
  }

  // Rank by relevance (desc), tie-break by smaller payload for determinism.
  const ranked = [...scored].sort((a, b) => b.score - a.score || a.tokens - b.tokens);

  const relevant = ranked.filter((entry) => entry.score >= minRelevance && entry.score > 0);
  const pool = relevant.length > 0 ? relevant : ranked;

  const enabledNames: string[] = [];
  let enabledTokens = 0;

  for (const entry of pool) {
    if (options?.tokenBudget !== undefined && enabledTokens + entry.tokens > options.tokenBudget) {
      // Budget would be exceeded by this (lower-relevance) tool — stop admitting.
      if (enabledNames.length > 0) {
        break;
      }
      // Nothing admitted yet: allow the single most-relevant tool through even
      // if it alone exceeds the budget, so gating never strands the agent.
    }
    if (options?.maxTools !== undefined && enabledNames.length >= options.maxTools) {
      break;
    }

    enabledNames.push(entry.name);
    enabledTokens += entry.tokens;
  }

  const enabledSet = new Set(enabledNames);
  const disabledNames = ranked.map((entry) => entry.name).filter((name) => !enabledSet.has(name));

  return {
    enabled: enabledNames,
    disabled: disabledNames,
    scores,
    estimatedEnabledTokens: enabledTokens,
    estimatedSavedTokens: Math.max(0, totalTokens - enabledTokens),
  };
};
