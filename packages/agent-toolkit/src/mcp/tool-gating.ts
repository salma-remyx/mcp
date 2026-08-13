import { z, ZodRawShape } from 'zod';
import { zodToJsonSchema } from 'zod-to-json-schema';
import { Tool } from '../core/tool';

/**
 * Dynamic tool gating and lazy schema loading — a target-native adaptation of the
 * "Tool Attention" idea (arXiv:2604.21816) for cutting the per-turn "MCP/Tools Tax".
 *
 * The monday.com MCP server eagerly advertises the full JSON schema of every registered
 * tool on every turn. With ~77 tools that is a large, mostly-irrelevant payload that
 * re-inflates the key-value cache each turn. This module provides three pieces to cut it:
 *
 *   1. A tax METER (`estimateToolTokens`) that quantifies each tool's per-turn schema cost.
 *   2. A relevance GATE (`selectToolsForTurn`) that picks the tools worth exposing for a
 *      given turn within a token budget and defers the rest.
 *   3. A lazy SCHEMA LOADER (`materializeToolSchema`) so a deferred tool's full schema is
 *      materialized on demand instead of every turn.
 *
 * Mode 2 (adapted port): the paper's learned tool-attention gate is replaced by a
 * parameter-free relevance proxy (query <-> tool name+description term overlap). The core
 * mechanisms — per-turn gating under a token budget and deferred schema materialization —
 * are kept at full fidelity. The paper's bespoke eval/benchmark framework is intentionally
 * out of scope (evaluation belongs in a downstream PR).
 */

/** Rough chars-per-token heuristic used to meter schema cost. */
const CHARS_PER_TOKEN = 4;

/** Default floor: always expose at least this many (most-relevant) tools per turn. */
const DEFAULT_MIN_TOOLS = 3;

const STOPWORDS = new Set<string>([
  'the', 'a', 'an', 'and', 'or', 'to', 'of', 'for', 'in', 'on', 'at', 'by', 'with', 'from', 'into',
  'my', 'your', 'i', 'you', 'is', 'are', 'be', 'this', 'that', 'it',
]);

/** Options for {@link selectToolsForTurn}. */
export interface GateOptions {
  /** Approximate-token cap for the schemas exposed this turn. Undefined = no cap (passthrough). */
  tokenBudget?: number;
  /** Floor: always expose at least this many most-relevant tools. Defaults to 3. */
  minTools?: number;
}

/** Result of gating a turn: which tools to expose vs. defer, plus the measured tax cut. */
export interface GateDecision {
  /** Tools to expose this turn (full schemas), ranked most-relevant first. */
  enabled: Tool<any, any>[];
  /** Tools held back this turn — their schemas are NOT materialized (lazy). */
  deferred: Tool<any, any>[];
  enabledNames: string[];
  deferredNames: string[];
  /** MCP Tax if every tool were eagerly exposed. */
  totalSchemaTokens: number;
  /** MCP Tax under the gate (exposed tools only). */
  gatedSchemaTokens: number;
  /** Tax eliminated this turn: `totalSchemaTokens - gatedSchemaTokens`. */
  tokensSaved: number;
}

/**
 * Split text into a normalized set of meaningful terms: lowercased alphanumeric runs of
 * length >= 2 with glue words removed. Feeds the relevance proxy.
 */
export function tokenize(text: string): Set<string> {
  const terms = new Set<string>();
  if (!text) {
    return terms;
  }
  for (const raw of text.toLowerCase().split(/[^a-z0-9]+/)) {
    if (raw.length >= 2 && !STOPWORDS.has(raw)) {
      terms.add(raw);
    }
  }
  return terms;
}

/**
 * Materialize a tool's full JSON schema on demand (lazy schema loading). Returns undefined
 * for tools without an input schema. Robust to non-Zod shapes via a JSON fallback so the
 * meter never throws on unusual schema objects.
 */
export function materializeToolSchema(tool: Tool<any, any>): Record<string, unknown> | undefined {
  const schema = tool.getInputSchema() as ZodRawShape | undefined;
  if (!schema) {
    return undefined;
  }
  try {
    return zodToJsonSchema(z.object(schema)) as Record<string, unknown>;
  } catch {
    try {
      return JSON.parse(JSON.stringify(schema)) as Record<string, unknown>;
    } catch {
      return undefined;
    }
  }
}

/**
 * Meter the approximate token cost of a tool's per-turn schema payload (its slice of the
 * MCP Tax): the serialized name + description + lazily-materialized JSON schema.
 */
export function estimateToolTokens(tool: Tool<any, any>): number {
  const schema = materializeToolSchema(tool);
  const text = `${tool.name} ${tool.getDescription() ?? ''} ${schema ? JSON.stringify(schema) : ''}`;
  return Math.ceil(text.length / CHARS_PER_TOKEN);
}

/**
 * Parameter-free relevance proxy substituting the paper's learned tool-attention gate.
 * Scores how strongly a tool's name + description overlap the query terms, with a small
 * bonus when query terms hit the tool's name directly. Returns 0 for an empty query.
 */
export function scoreToolRelevance(tool: Tool<any, any>, query: string): number {
  const queryTerms = tokenize(query);
  if (queryTerms.size === 0) {
    return 0;
  }
  const nameTerms = tokenize(tool.name);
  const descriptionTerms = tokenize(tool.getDescription() ?? '');
  const toolTerms = new Set<string>([...nameTerms, ...descriptionTerms]);

  let matched = 0;
  let nameHits = 0;
  for (const term of queryTerms) {
    if (toolTerms.has(term)) {
      matched++;
    }
    if (nameTerms.has(term)) {
      nameHits++;
    }
  }
  const coverage = matched / queryTerms.size;
  const nameBonus = (nameHits / queryTerms.size) * 0.5;
  return coverage + nameBonus;
}

/**
 * Core gate: pick the tools worth exposing for a turn.
 *
 * Tools are ranked by relevance to the query and greedily admitted until the token budget
 * is exhausted, always keeping at least `minTools` exposed (so the agent is never left with
 * nothing when query terms happen to overlap no tool). The rest are deferred — their
 * schemas are not materialized this turn (lazy schema loading). With no query the gate is a
 * passthrough (all tools enabled, zero tax cut), preserving the current eager behavior.
 */
export function selectToolsForTurn(
  tools: Tool<any, any>[],
  query: string,
  options?: GateOptions,
): GateDecision {
  const totalSchemaTokens = tools.reduce((sum, tool) => sum + estimateToolTokens(tool), 0);

  if (!query || tools.length === 0) {
    return {
      enabled: [...tools],
      deferred: [],
      enabledNames: tools.map((tool) => tool.name),
      deferredNames: [],
      totalSchemaTokens,
      gatedSchemaTokens: totalSchemaTokens,
      tokensSaved: 0,
    };
  }

  const minTools = Math.max(0, options?.minTools ?? DEFAULT_MIN_TOOLS);
  const tokenBudget = options?.tokenBudget;

  const scored = tools
    .map((tool) => ({
      tool,
      relevance: scoreToolRelevance(tool, query),
      tokens: estimateToolTokens(tool),
    }))
    .sort((a, b) => b.relevance - a.relevance || a.tokens - b.tokens);

  const enabledSet = new Set<string>();
  const enabled: Tool<any, any>[] = [];
  let gatedSchemaTokens = 0;
  for (const item of scored) {
    const withinBudget = tokenBudget === undefined || gatedSchemaTokens + item.tokens <= tokenBudget;
    if (enabled.length < minTools || withinBudget) {
      enabled.push(item.tool);
      enabledSet.add(item.tool.name);
      gatedSchemaTokens += item.tokens;
    }
  }

  const deferred = tools.filter((tool) => !enabledSet.has(tool.name));

  return {
    enabled,
    deferred,
    enabledNames: enabled.map((tool) => tool.name),
    deferredNames: deferred.map((tool) => tool.name),
    totalSchemaTokens,
    gatedSchemaTokens,
    tokensSaved: Math.max(0, totalSchemaTokens - gatedSchemaTokens),
  };
}
