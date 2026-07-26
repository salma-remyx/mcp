/**
 * Dynamic tool gating — per-turn relevance selection over the registered tool
 * surface.
 *
 * Adapted (Mode 2) from "Tool Attention Is All You Need: Dynamic Tool Gating
 * and Lazy Schema Loading for Eliminating the MCP/Tools Tax in Scalable
 * Agentic Workflows" (arXiv:2604.21816). The paper's headline mechanism is a
 * *per-turn tool-attention* gate that prunes which tool schemas an agent sees
 * each turn, eliminating the eager-schema "MCP tax" (reported ~10k–60k tokens
 * per turn in multi-server deployments). This module keeps that core mechanism
 * at full fidelity while substituting one auxiliary component:
 *
 *   - SUBSTITUTED: the paper's *learned* tool-attention ranker (a model trained
 *     on tool-use traces) is replaced by a parameter-free lexical-relevance
 *     proxy: a weighted token-overlap score between the turn's query and each
 *     tool's name / description / argument vocabulary. No weights are learned.
 *
 *   - INTENTIONALLY OUT OF SCOPE: the paper's separate lazy-schema-loading
 *     transport and its benchmark suite. Gating already removes non-relevant
 *     schemas from the active per-turn surface, which is where the token
 *     savings accrue; the transport belongs in a downstream change.
 *
 * The scoring is deterministic and dependency-free so it can be unit-tested
 * and reasoned about without a model in the loop.
 */

import { Tool } from '../core/tool';

/**
 * Structural view of a tool needed to score it. `Tool<any, any>` satisfies it,
 * but tests can pass minimal objects without building a full tool.
 */
export interface GatableTool {
  name: string;
  getDescription(): string;
  getInputSchema(): any;
}

/** Options controlling how the gate prunes the tool surface. */
export interface GateOptions {
  /** Keep at most this many tools (highest-scoring first). Unset = keep all relevant. */
  topK?: number;
  /** Keep only tools scoring at least this (0–1). Default 0 ⇒ any overlap. */
  minScore?: number;
}

/** A tool that survived the gate, with its relevance score. */
export interface GatedTool {
  name: string;
  score: number;
}

/** Result of wiring the gate through the toolkit's enable/disable surface. */
export interface ToolGateResult {
  query: string;
  /** Tool names that remained enabled after gating. */
  kept: string[];
  /** Tool names disabled by this gate. */
  disabled: string[];
  /** Per-tool relevance score for every kept tool. */
  scores: Record<string, number>;
}

// Common English stopwords plus filler words that carry no tool signal.
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
  'have',
  'i',
  'in',
  'is',
  'it',
  'its',
  'of',
  'on',
  'or',
  'that',
  'the',
  'this',
  'to',
  'want',
  'wants',
  'need',
  'needs',
  'please',
  'me',
  'my',
  'we',
  'with',
  'all',
  'get',
  'do',
  'can',
  'you',
  'show',
  'use',
  'using',
  'into',
]);

// Name-token matches are the strongest relevance signal (the tool's identity),
// description matches are weaker, and argument-name matches weaker still.
const NAME_WEIGHT = 3;
const DESC_WEIGHT = 1;
const ARG_WEIGHT = 0.5;

/**
 * Split text into normalized query/term tokens. Handles camelCase, kebab-case,
 * snake_case, and path separators, then lowercases, drops stopwords and tokens
 * shorter than two characters.
 */
export function tokenize(text: string): string[] {
  if (!text) return [];
  return text
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .replace(/[_\-/.]+/g, ' ')
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((token) => token.length > 1 && !STOPWORDS.has(token));
}

function addAll(set: Set<string>, tokens: string[]): void {
  for (const token of tokens) set.add(token);
}

/** Tokenize a tool into weighted vocabulary buckets used for overlap scoring. */
export function getToolVocabulary(tool: GatableTool): {
  name: Set<string>;
  description: Set<string>;
  args: Set<string>;
} {
  const name = new Set<string>();
  const description = new Set<string>();
  const args = new Set<string>();

  addAll(name, tokenize(tool.name));
  addAll(description, tokenize(tool.getDescription()));

  const schema = tool.getInputSchema();
  if (schema && typeof schema === 'object') {
    for (const [key, node] of Object.entries(schema)) {
      addAll(args, tokenize(key));
      const desc = (node as any)?.description;
      if (typeof desc === 'string') {
        addAll(args, tokenize(desc));
      }
    }
  }

  return { name, description, args };
}

/**
 * Score how relevant a tool is to a query, in [0, 1]. A tool scores 0 when it
 * shares no tokens with the query; 1 when every query token appears in the
 * tool's name. Description-only and argument-only matches score lower.
 *
 * Each unique query token contributes up to NAME_WEIGHT to the numerator; the
 * score is normalized by the maximum possible numerator (every token hitting at
 * name level) so the result is length-invariant.
 */
export function scoreToolRelevance(query: string, tool: GatableTool): number {
  const queryTokens = new Set(tokenize(query));
  if (queryTokens.size === 0) return 0;

  const vocab = getToolVocabulary(tool);
  let numerator = 0;
  for (const token of queryTokens) {
    if (vocab.name.has(token)) numerator += NAME_WEIGHT;
    else if (vocab.description.has(token)) numerator += DESC_WEIGHT;
    else if (vocab.args.has(token)) numerator += ARG_WEIGHT;
  }

  const maxNumerator = queryTokens.size * NAME_WEIGHT;
  return maxNumerator === 0 ? 0 : numerator / maxNumerator;
}

/**
 * Gate a set of tools down to those relevant to a query. Tools with no token
 * overlap (score 0) are always dropped; the rest are ranked by relevance and
 * optionally trimmed to `topK` / a `minScore` floor. Ties break by name for
 * deterministic output.
 */
export function gateTools(
  query: string,
  tools: GatableTool[] | readonly Tool<any, any>[],
  options?: GateOptions,
): GatedTool[] {
  const minScore = options?.minScore ?? 0;
  const scored = tools
    .map((tool) => ({ name: tool.name, score: scoreToolRelevance(query, tool) }))
    .filter((gated) => gated.score > 0 && gated.score >= minScore)
    .sort((a, b) => (a.score !== b.score ? b.score - a.score : a.name.localeCompare(b.name)));

  const topK = options?.topK;
  return typeof topK === 'number' && topK >= 0 ? scored.slice(0, topK) : scored;
}
