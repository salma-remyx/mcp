import { Tool } from '../core/tool';

/**
 * Tool gating and lazy schema loading.
 *
 * Targets the per-turn "MCP/Tools tax" — the 10k–60k token cost of eagerly injecting
 * every tool's full JSON schema each turn — by (1) scoring each tool's relevance to the
 * current user intent and selecting only the relevant subset (gating), and (2) emitting a
 * compact tool index (name + short description + parameter names) that defers full schema
 * expansion until a tool is actually picked (lazy schema loading).
 *
 * Adapted from "Tool Attention Is All You Need: Dynamic Tool Gating and Lazy Schema
 * Loading for Eliminating the MCP/Tools Tax in Scalable Agentic Workflows"
 * (arXiv:2604.21816). The paper's learned Intent-Schema-Overlap estimator is replaced
 * here by a parameter-free token-overlap proxy over the same signals (tool name,
 * description, parameter names and descriptions); the paper's separate benchmark /
 * evaluation suite is intentionally out of scope.
 */

export interface ToolGatingOptions {
  /** Maximum number of tools to keep after gating. Defaults to 50. */
  maxTools?: number;
  /** Minimum relevance score (inclusive) required to keep a tool. Defaults to ~0. */
  minScore?: number;
  /** Truncate each tool's description to this many characters in the lazy index. Defaults to 120. */
  descriptionLimit?: number;
}

export interface LazyToolIndexEntry {
  name: string;
  description: string;
  paramNames: string[];
  score: number;
}

const DEFAULT_MIN_SCORE = 0.001;
const DEFAULT_MAX_TOOLS = 50;
const DEFAULT_DESCRIPTION_LIMIT = 120;

const STOPWORDS = new Set([
  'the',
  'a',
  'an',
  'and',
  'or',
  'of',
  'to',
  'in',
  'on',
  'for',
  'with',
  'is',
  'are',
  'be',
  'by',
  'this',
  'that',
  'it',
  'as',
  'at',
  'from',
  'my',
  'our',
  'i',
  'please',
  'can',
  'you',
  'your',
  'do',
  'how',
  'what',
  'which',
  'want',
  'need',
  'me',
  'using',
  'use',
  'all',
  'new',
]);

/** Lowercase, split on non-alphanumeric boundaries, drop stopwords and 1-char tokens. */
function tokenize(text: string): string[] {
  if (!text) return [];
  return text
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((token) => token.length > 1 && !STOPWORDS.has(token));
}

/** Split a snake_case / kebab-case tool name into its constituent terms. */
function tokenizeToolName(name: string): string[] {
  return tokenize(name.replace(/[_-]+/g, ' '));
}

/** Best-effort description text for a single Zod schema field (a ZodRawShape value). */
function fieldDescription(field: any): string {
  if (!field) return '';
  if (typeof field.description === 'string') return field.description;
  // Unwrap optional / defaulted / nullable wrappers, then re-check.
  if (typeof field.unwrap === 'function') {
    const inner = field.unwrap();
    if (inner && typeof inner.description === 'string') return inner.description;
  }
  return '';
}

/** Parameter names declared by a tool's input schema. */
function paramNamesOf(tool: Tool<any, any>): string[] {
  const schema = tool.getInputSchema();
  return schema && typeof schema === 'object' ? Object.keys(schema) : [];
}

/** Token set representing a tool's surface — the "schema" in Intent-Schema-Overlap. */
function toolSurfaceTokens(tool: Tool<any, any>): Set<string> {
  const tokens: string[] = [...tokenizeToolName(tool.name), ...tokenize(tool.getDescription())];
  const schema = tool.getInputSchema();
  if (schema && typeof schema === 'object') {
    for (const [paramName, field] of Object.entries(schema)) {
      tokens.push(...tokenize(paramName));
      tokens.push(...tokenize(fieldDescription(field)));
    }
  }
  return new Set(tokens);
}

/**
 * Relevance of a tool to a user query — a parameter-free proxy for the paper's learned
 * Intent-Schema-Overlap estimator. Weighted count of query terms present in the tool's
 * surface (name matches weighted higher), normalized by query length. Returns 0 when the
 * query carries no signal.
 */
export function scoreToolRelevance(query: string, tool: Tool<any, any>): number {
  const queryTokens = tokenize(query);
  if (queryTokens.length === 0) return 0;

  const nameTokens = new Set(tokenizeToolName(tool.name));
  const surfaceTokens = toolSurfaceTokens(tool);

  let score = 0;
  for (const term of queryTokens) {
    if (nameTokens.has(term)) {
      score += 2; // name overlap is the strongest intent signal
    } else if (surfaceTokens.has(term)) {
      score += 1;
    }
  }
  return score / queryTokens.length;
}

/** Rank tools by descending relevance to the query; ties preserve insertion order. */
export function rankToolsByRelevance(
  query: string,
  tools: Tool<any, any>[],
): Array<{ tool: Tool<any, any>; score: number }> {
  return tools.map((tool) => ({ tool, score: scoreToolRelevance(query, tool) })).sort((a, b) => b.score - a.score);
}

/** Gated subset of tools relevant to the query. No gating is applied when the query is empty. */
export function selectRelevantTools(
  query: string,
  tools: Tool<any, any>[],
  options?: ToolGatingOptions,
): Tool<any, any>[] {
  if (tokenize(query).length === 0) return tools; // no intent → no gating

  const minScore = options?.minScore ?? DEFAULT_MIN_SCORE;
  const maxTools = options?.maxTools ?? DEFAULT_MAX_TOOLS;

  return rankToolsByRelevance(query, tools)
    .filter((entry) => entry.score >= minScore)
    .slice(0, maxTools)
    .map((entry) => entry.tool);
}

/**
 * Compact per-tool index entries (phase 1 of lazy schema loading): name, a truncated
 * description, parameter names, and the relevance score. Full JSON schemas are omitted —
 * they are expanded only on demand for the gated subset.
 */
export function buildLazyToolIndex(
  query: string,
  tools: Tool<any, any>[],
  options?: ToolGatingOptions,
): LazyToolIndexEntry[] {
  const limit = options?.descriptionLimit ?? DEFAULT_DESCRIPTION_LIMIT;
  return rankToolsByRelevance(query, tools).map(({ tool, score }) => {
    const description = tool.getDescription();
    return {
      name: tool.name,
      description: description.length > limit ? `${description.slice(0, limit).trimEnd()}…` : description,
      paramNames: paramNamesOf(tool),
      score,
    };
  });
}
