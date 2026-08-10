import { z } from 'zod';
import { zodToJsonSchema } from 'zod-to-json-schema';
import { Tool } from '../core/tool';
import { ToolkitManager } from '../core/tools/platform-api-tools/manage-tools-tool';

/**
 * Intent-routed tool gating + lazy schema loading.
 *
 * Adapted from "Tool Attention Is All You Need: Dynamic Tool Gating and
 * Lazy Schema Loading for Eliminating the MCP/Tools Tax in Scalable
 * Agentic Workflows" (arXiv:2604.21816). The paper shows that eagerly
 * injecting every tool's schema on every agent turn costs ~10k–60k
 * tokens (the "MCP Tax"); it proposes (a) dynamic tool gating — expose
 * only the intent-relevant subset per turn — and (b) lazy schema
 * loading — materialize JSON schemas only for the gated subset.
 *
 * This module keeps both core mechanisms at full fidelity and drives the
 * repo's existing DynamicToolManager (which already supports MCP
 * `tools/listChanged`), so a gated turn re-uses the protocol path that
 * `MondayAgentToolkit` already speaks.
 *
 * Mode-2 substitutions (auxiliary components swapped for parameter-free,
 * target-native equivalents — see PR description):
 *   - The paper's *learned* tool-attention / mutual-information estimator
 *     is replaced by a parameter-free intent->tool relevance scorer: a
 *     weighted term-overlap heuristic over each tool's name + description
 *     + annotation title (name hits weighted above description hits).
 *   - The paper's bespoke benchmark/eval harness is intentionally cut;
 *     evaluation belongs in a downstream PR. A lightweight token-payload
 *     estimate (`estimatePayloadTokens`) is exposed so callers can observe
 *     the MCP-Tax reduction directly.
 */

/** Rough chars-per-token factor for the payload estimate. */
const CHARS_PER_TOKEN = 4;

const DEFAULT_MAX_TOOLS = 8;
const DEFAULT_MIN_SCORE = 1;
const DEFAULT_MIN_TOOLS = 1;

/** Common English filler words ignored when tokenizing an intent. */
const STOPWORDS = new Set([
  'the',
  'and',
  'for',
  'with',
  'that',
  'this',
  'from',
  'your',
  'you',
  'are',
  'was',
  'but',
  'not',
  'how',
  'what',
  'when',
  'who',
  'all',
  'can',
  'get',
  'use',
  'using',
  'into',
  'our',
  'out',
  'has',
  'have',
  'will',
  'would',
  'could',
  'should',
  'want',
  'need',
  'please',
  'about',
]);

export interface ToolGatingOptions {
  /** Maximum tools exposed per turn. Default 8. */
  maxTools?: number;
  /** Minimum relevance score required to include a tool. Default 1. */
  minScore?: number;
  /** Tool names always exposed regardless of score (e.g. `manage_tools`). */
  alwaysInclude?: string[];
  /** Floor on gated set size before the safety fallback fires. Default 1. */
  minTools?: number;
  /** When the gate selects fewer than `minTools`, expose the full catalog. Default true. */
  fallbackToAll?: boolean;
}

export interface GatingResult {
  /** Names of the tools selected for this turn, in ranked order. */
  selected: string[];
  /** Names of tools disabled because they fell outside the gate. */
  disabled: string[];
  /** Lazy JSON schemas for the selected tools only (name -> schema). */
  schemas: Record<string, unknown>;
  /** Estimated serialized payload of the full catalog, in tokens. */
  tokensBefore: number;
  /** Estimated serialized payload of the gated subset, in tokens. */
  tokensAfter: number;
  /** `tokensBefore - tokensAfter` (never negative). */
  tokensSaved: number;
  /** True when the safety fallback exposed the full catalog. */
  fellBack: boolean;
}

/**
 * Split free text into normalized search terms: lowercase alphanumeric
 * runs, dropping stopwords and runs of length <= 2.
 */
export function tokenizeIntent(text: string): string[] {
  if (!text) return [];
  const terms = text
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((term) => term.length > 2 && !STOPWORDS.has(term));
  return Array.from(new Set(terms));
}

/**
 * Relevance of a tool to an intent. Parameter-free proxy for the paper's
 * learned tool-attention weight: name-token matches count 3x description
 * hits, reflecting that a tool's name is its strongest intent signal.
 */
export function scoreTool(intent: string, tool: Tool<any, any>): number {
  const terms = tokenizeIntent(intent);
  if (terms.length === 0) return 0;

  const nameTokens = new Set(
    (tool.name ?? '')
      .toLowerCase()
      .split(/[^a-z0-9]+/)
      .filter(Boolean),
  );
  const haystack = `${(tool.annotations as { title?: string } | undefined)?.title ?? ''} ${
    tool.getDescription?.() ?? ''
  }`.toLowerCase();

  let score = 0;
  for (const term of terms) {
    if (nameTokens.has(term)) {
      score += 3;
    } else if (haystack.includes(term)) {
      score += 1;
    }
  }
  return score;
}

/** Rank all tools by descending relevance to the intent. */
export function rankTools(intent: string, tools: Tool<any, any>[]): Array<{ tool: Tool<any, any>; score: number }> {
  return tools.map((tool) => ({ tool, score: scoreTool(intent, tool) })).sort((a, b) => b.score - a.score);
}

/**
 * Select the gated subset for an intent: tools scoring at or above
 * `minScore` (plus any `alwaysInclude` names), capped at `maxTools`.
 */
export function selectTools(
  intent: string,
  tools: Tool<any, any>[],
  options: ToolGatingOptions = {},
): Tool<any, any>[] {
  const { maxTools = DEFAULT_MAX_TOOLS, minScore = DEFAULT_MIN_SCORE, alwaysInclude = [] } = options;
  const always = new Set(alwaysInclude);

  const picked: Tool<any, any>[] = [];
  for (const { tool, score } of rankTools(intent, tools)) {
    if (picked.length >= maxTools) break;
    if (score >= minScore || always.has(tool.name)) {
      picked.push(tool);
    }
  }

  // Ensure safety-net tools that exist are present even past the cap.
  for (const tool of tools) {
    if (always.has(tool.name) && !picked.some((existing) => existing.name === tool.name)) {
      picked.push(tool);
    }
  }
  return picked;
}

/**
 * Lazily serialize a single tool's input schema to JSON Schema, mirroring
 * `MondayAgentToolkit.getSchemaForTool`'s JSON path. Returns undefined for
 * tools with no input schema. This is the "lazy" half of the mechanism:
 * the caller only pays the serialization cost for tools inside the gate.
 */
export function serializeToolSchema(tool: Tool<any, any>): unknown {
  const inputSchema = tool.getInputSchema?.();
  if (!inputSchema) return undefined;
  try {
    return zodToJsonSchema(z.object(inputSchema));
  } catch {
    return undefined;
  }
}

/** Estimate the per-turn token payload of a tool set (name + description + schema). */
export function estimatePayloadTokens(tools: Tool<any, any>[]): number {
  let chars = 0;
  for (const tool of tools) {
    chars += (tool.name ?? '').length;
    chars += (tool.getDescription?.() ?? '').length;
    const schema = serializeToolSchema(tool);
    if (schema) chars += JSON.stringify(schema).length;
  }
  return Math.round(chars / CHARS_PER_TOKEN);
}

/**
 * Routes a single turn: gates the catalog to the intent-relevant subset,
 * applies that decision to the existing {@link ToolkitManager} (so the
 * MCP server's `tools/listChanged` machinery re-emits the slimmed list),
 * and returns lazy schemas plus the MCP-Tax reduction estimate.
 */
export class ToolGatingRouter {
  constructor(
    private readonly manager: ToolkitManager,
    private readonly options: ToolGatingOptions = {},
  ) {}

  route(intent: string, tools: Tool<any, any>[]): GatingResult {
    const {
      maxTools = DEFAULT_MAX_TOOLS,
      minScore = DEFAULT_MIN_SCORE,
      alwaysInclude = [],
      minTools = DEFAULT_MIN_TOOLS,
      fallbackToAll = true,
    } = this.options;

    let picked = selectTools(intent, tools, { maxTools, minScore, alwaysInclude });
    let fellBack = false;
    // Safety net: never starve the agent. If the gate is empty/unconfident,
    // fall back to the eager (full) catalog rather than exposing nothing.
    if (picked.length < minTools && fallbackToAll) {
      picked = tools.slice();
      fellBack = true;
    }

    const selectedNames = new Set(picked.map((tool) => tool.name));
    const disabled: string[] = [];
    for (const tool of tools) {
      if (selectedNames.has(tool.name)) {
        this.manager.enableTool(tool.name);
      } else {
        this.manager.disableTool(tool.name);
        disabled.push(tool.name);
      }
    }

    const schemas: Record<string, unknown> = {};
    for (const tool of picked) {
      schemas[tool.name] = serializeToolSchema(tool);
    }

    const tokensBefore = estimatePayloadTokens(tools);
    const tokensAfter = estimatePayloadTokens(picked);

    return {
      selected: picked.map((tool) => tool.name),
      disabled,
      schemas,
      tokensBefore,
      tokensAfter,
      tokensSaved: Math.max(0, tokensBefore - tokensAfter),
      fellBack,
    };
  }
}
