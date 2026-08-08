import { DynamicToolManager } from './dynamic-tool-manager';
import { Tool } from '../core/tool';

/**
 * Per-turn, intent-conditioned tool gating ("Tool Attention") for the MCP toolkit.
 *
 * MCP eagerly injects every registered tool's JSON schema into the model context on
 * every turn — the "MCP Tax" / "Tools Tax", reported in the wild at roughly 10k–60k
 * tokens for multi-tool servers. This module trims that tax by gating the toolset per
 * turn: only the tools whose name/description overlap the current user intent are kept
 * "on" (advertised via tools/list); the rest are disabled, so their schemas stop being
 * resident.
 *
 * It composes with the toolkit's existing primitives rather than replacing them:
 *   - input is the output of the static `getFilteredToolInstances` config filter
 *     (mode/include/exclude/readOnly), so permission/mode filtering still wins;
 *   - the actuator is the existing `DynamicToolManager` (`enableTool` / `disableTool`),
 *     whose enabled set already drives the MCP `tools/list` response.
 *
 * Adapted from "Tool Attention Is All You Need: Dynamic Tool Gating and Lazy Schema
 * Loading for Eliminating the MCP/Tools Tax in Scalable Agentic Workflows"
 * (arxiv 2604.21816). The paper's per-turn intent-conditioned gating is reproduced at
 * full fidelity (gate by intent → drive enable/disable → shrink tools/list). Its LEARNED
 * intent router is replaced by a parameter-free vocab-overlap proxy, so the policy needs
 * no model, training data, or extra inference call. The paper's separate multi-server
 * benchmark suite is intentionally out of scope here.
 */

const NAME_WEIGHT = 2;
const DESC_WEIGHT = 1;

const DEFAULT_STOPWORDS = new Set<string>([
  'a',
  'an',
  'the',
  'and',
  'or',
  'but',
  'of',
  'to',
  'in',
  'on',
  'for',
  'at',
  'by',
  'with',
  'from',
  'into',
  'is',
  'are',
  'be',
  'this',
  'that',
  'it',
  'as',
  'my',
  'our',
  'i',
  'we',
  'me',
  'us',
  'please',
  'want',
  'need',
  'using',
  'use',
  'do',
  'does',
  'new',
  'all',
]);

export interface ToolAttentionOptions {
  /** Max tools to keep advertised per turn. Others are gated off (lazy schema). Default 5. */
  maxTools?: number;
  /** Minimum [0,1] relevance score required to keep a tool on. Default 0 (overlap only). */
  minScore?: number;
  /** Tool names that must stay on regardless of score (e.g. the manage-tools tool). */
  alwaysOn?: string[];
  /** Override the default English stopword set. */
  stopwords?: Set<string>;
}

export interface ToolRelevanceScore {
  name: string;
  score: number;
}

export interface ToolAttentionResult {
  /** Tool names kept advertised this turn. */
  enabled: string[];
  /** Tool names gated off (schemas no longer resident) this turn. */
  disabled: string[];
  /** Per-tool relevance scores for observability. */
  scores: ToolRelevanceScore[];
}

/** A tool (or a stand-in) carrying the text the gating policy matches against. */
export interface GatableTool {
  name: string;
  description: string;
}

const normalize = (token: string): string => {
  // Crude plural/stem normalization so "items" ~ "item", "boards" ~ "board".
  const lowered = token.toLowerCase();
  return lowered.endsWith('s') && lowered.length > 3 ? lowered.slice(0, -1) : lowered;
};

const tokenize = (text: string, stopwords: Set<string>): string[] => {
  const seen = new Set<string>();
  const tokens: string[] = [];
  for (const raw of text.toLowerCase().split(/[^a-z0-9]+/)) {
    if (raw.length < 2) continue;
    const tok = normalize(raw);
    if (tok.length < 2 || stopwords.has(raw) || stopwords.has(tok)) continue;
    if (seen.has(tok)) continue;
    seen.add(tok);
    tokens.push(tok);
  }
  return tokens;
};

const toGatable = (tool: Tool<any, any>): GatableTool => ({
  name: tool.name,
  description: tool.getDescription(),
});

/**
 * Parameter-free relevance of a tool to an intent, in [0,1].
 * Intent tokens that appear in the tool NAME count NAME_WEIGHT; tokens only in the
 * DESCRIPTION count DESC_WEIGHT. This vocab-overlap proxy stands in for the paper's
 * learned intent-conditioned attention weights — same gating signal, no model.
 */
export const scoreToolRelevance = (intent: string, tool: GatableTool, options?: ToolAttentionOptions): number => {
  const stopwords = options?.stopwords ?? DEFAULT_STOPWORDS;
  const intentTokens = tokenize(intent, stopwords);
  if (intentTokens.length === 0) return 0;

  const nameTokens = new Set(tokenize(tool.name, stopwords));
  const descTokens = new Set(tokenize(tool.description, stopwords));

  let weightedHits = 0;
  for (const token of intentTokens) {
    if (nameTokens.has(token)) {
      weightedHits += NAME_WEIGHT;
    } else if (descTokens.has(token)) {
      weightedHits += DESC_WEIGHT;
    }
  }
  return Math.min(1, weightedHits / (NAME_WEIGHT * intentTokens.length));
};

/**
 * Lazy-schema selection: the subset of `tools` to advertise for `intent` this turn.
 * Compose with the static `getFilteredToolInstances` output — run the config filter
 * first, then gate by intent. `alwaysOn` tools are force-included without consuming
 * the `maxTools` budget.
 */
export const selectTools = <T extends GatableTool>(
  intent: string,
  tools: T[],
  options?: ToolAttentionOptions,
): { selected: T[]; gatedOff: T[]; scores: ToolRelevanceScore[] } => {
  const maxTools = options?.maxTools ?? 5;
  const minScore = options?.minScore ?? 0;
  const alwaysOn = new Set(options?.alwaysOn ?? []);

  const scored = tools.map((tool) => ({
    tool,
    score: alwaysOn.has(tool.name) ? 1 : scoreToolRelevance(intent, tool, options),
  }));
  scored.sort((a, b) => b.score - a.score);

  // alwaysOn tools are force-selected and do NOT consume the maxTools budget.
  const pinned = scored.filter((s) => alwaysOn.has(s.tool.name));
  const candidates = scored.filter((s) => !alwaysOn.has(s.tool.name)); // still sorted by score desc

  const selected: T[] = pinned.map((s) => s.tool);
  const gatedOff: T[] = [];

  for (const { tool, score } of candidates) {
    // Only non-pinned selections count toward the budget.
    if (selected.length - pinned.length >= maxTools || score < minScore) {
      gatedOff.push(tool);
      continue;
    }
    selected.push(tool);
  }

  const scores: ToolRelevanceScore[] = scored.map(({ tool, score }) => ({ name: tool.name, score }));
  return { selected, gatedOff, scores };
};

/**
 * Apply per-turn Tool Attention to a live {@link DynamicToolManager}: enable the
 * intent-relevant tools and disable the rest. Because the manager's enabled set drives
 * the MCP `tools/list` response, gated-off tools' schemas drop out of context for the
 * turn — the lazy-schema / MCP-Tax reduction. Intended as a one-line per-turn hook:
 *
 *   applyToolAttention(toolkit.dynamicToolManager, userIntent, { maxTools: 5 });
 *
 * Only tools whose enabled-state actually changed appear in `enabled` / `disabled`.
 */
export const applyToolAttention = (
  manager: DynamicToolManager,
  intent: string,
  options?: ToolAttentionOptions,
): ToolAttentionResult => {
  const registered = manager.getAllDynamicTools();
  const tools: GatableTool[] = [];
  registered.forEach((entry, name) => {
    if (entry?.instance) {
      tools.push(toGatable(entry.instance));
    } else {
      tools.push({ name, description: name });
    }
  });

  const { selected, scores } = selectTools(intent, tools, options);
  const selectedNames = new Set(selected.map((t) => t.name));

  const enabled: string[] = [];
  const disabled: string[] = [];

  registered.forEach((_entry, name) => {
    const wasOn = manager.isToolEnabled(name);
    if (selectedNames.has(name)) {
      manager.enableTool(name);
      if (!wasOn) enabled.push(name);
    } else {
      manager.disableTool(name);
      if (wasOn) disabled.push(name);
    }
  });

  return { enabled, disabled, scores };
};
