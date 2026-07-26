import { Tool } from '../core/tool';

/**
 * Tool Attention — intent-driven tool gating.
 *
 * Adapted from "Tool Attention Is All You Need: Dynamic Tool Gating and Lazy
 * Schema Loading for Eliminating the MCP/Tools Tax in Scalable Agentic
 * Workflows" (arXiv:2604.21816). The paper's core mechanism is
 * Intent-Schema-Overlap gating: score each tool by the overlap between the
 * user's intent and the tool's schema, then gate ON only the top-K tools per
 * turn so the LLM never pays the per-turn schema tax for irrelevant tools.
 *
 * Mode 2 (adapted port). The paper's *learned* overlap estimator is replaced
 * here by a parameter-free vocabulary-overlap proxy — a normalized token-set
 * overlap (Sorensen-Dice) between the intent and the tool's name, description,
 * parameter names, parameter descriptions, and enum values. No embeddings, no
 * learned weights, no external model. The per-turn gating itself is applied by
 * the toolkit's `selectToolsForIntent`, which drives the existing
 * DynamicToolManager enable/disable hooks plus the MCP `listChanged`
 * capability the server already advertises — that realizes the paper's
 * lazy-loading effect natively, since only gated-on tools' full schemas are
 * exposed to the LLM in `tools/list`.
 *
 * Out of scope (intentional): a separate two-phase `listToolsPhase1` /
 * `getSchemaPhase2` transport protocol. The repo's `listChanged`-based
 * enable/disable already exposes only the gated tools, so a parallel phase
 * protocol would be redundant scaffolding rather than value.
 */

/** Lightweight "phase-1" view of a tool used for overlap scoring. */
export interface ToolManifestEntry {
  name: string;
  description: string;
  /** Tokenized name + description + every schema token (params, enum values). */
  schemaTokens: Set<string>;
}

export interface SelectToolsOptions {
  /** Maximum number of intent-matched tools to gate ON. Defaults to 10. */
  topK?: number;
  /** Minimum overlap score in [0, 1] required to keep a tool gated ON. Default 0.05. */
  minScore?: number;
  /** Tool names forced ON regardless of score (e.g. the management tool). */
  alwaysInclude?: string[];
}

export interface SelectedTool {
  name: string;
  score: number;
}

const STOPWORDS = new Set<string>([
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
  'was',
  'were',
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
  'he',
  'she',
  'my',
  'our',
  'your',
  'their',
  'me',
  'us',
  'them',
  'please',
  'want',
  'needs',
  'need',
  'can',
  'could',
  'should',
  'would',
  'will',
  'do',
  'does',
  'did',
  'using',
  'use',
  'used',
  'about',
  'into',
  'some',
  'all',
  'any',
  'how',
  'what',
  'which',
  'who',
  'whom',
  'when',
  'where',
  'why',
  'now',
]);

/**
 * Tokenize free text into a normalized, de-duplicated, stopword-free token list.
 */
export function tokenize(text: string): string[] {
  if (!text) {
    return [];
  }
  const raw = text.toLowerCase().split(/[^a-z0-9]+/);
  const seen = new Set<string>();
  const out: string[] = [];
  for (const token of raw) {
    if (!token || token.length < 2 || STOPWORDS.has(token)) {
      continue;
    }
    if (seen.has(token)) {
      continue;
    }
    seen.add(token);
    out.push(token);
  }
  return out;
}

/**
 * Tokenize an identifier (snake_case / camelCase / kebab-case) WITHOUT applying
 * the stopword filter, so the tool-name signal stays strong (e.g. "list", "get",
 * "create" matter for name overlap).
 */
export function tokenizeIdentifier(name: string): string[] {
  if (!name) {
    return [];
  }
  const parts = name
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .replace(/[_\-.]+/g, ' ')
    .toLowerCase()
    .split(/[^a-z0-9]+/);
  const seen = new Set<string>();
  const out: string[] = [];
  for (const token of parts) {
    if (!token || seen.has(token)) {
      continue;
    }
    seen.add(token);
    out.push(token);
  }
  return out;
}

/**
 * Extract schema tokens from a Zod raw shape: parameter names, each param's
 * `.describe()` text, and enum values. Defensive against arbitrary Zod shapes.
 */
function extractZodTokens(shape: unknown): string[] {
  const tokens: string[] = [];
  if (!shape || typeof shape !== 'object') {
    return tokens;
  }
  for (const [key, zodType] of Object.entries(shape as Record<string, any>)) {
    tokens.push(...tokenizeIdentifier(key));
    const z = zodType as any;
    if (typeof z?.description === 'string') {
      tokens.push(...tokenize(z.description));
    }
    const options = z?.options ?? z?._def?.values ?? z?._def?.innerType?.options;
    if (Array.isArray(options)) {
      for (const option of options) {
        if (typeof option === 'string') {
          tokens.push(...tokenize(option));
        }
      }
    }
  }
  return tokens;
}

/**
 * Build a compact manifest entry from a Tool: name + description + schema
 * tokens. This is the cheap "phase-1" view used for overlap scoring, never the
 * full JSON schema.
 */
export function buildToolManifest(tool: Tool<any, any>): ToolManifestEntry {
  const schemaTokens = new Set<string>([
    ...tokenizeIdentifier(tool.name),
    ...tokenize(tool.getDescription?.() ?? ''),
    ...extractZodTokens(tool.getInputSchema?.()),
  ]);
  return {
    name: tool.name,
    description: tool.getDescription?.() ?? '',
    schemaTokens,
  };
}

/**
 * Sorensen-Dice coefficient over two token sets, in [0, 1].
 */
function diceOverlap(a: Set<string>, b: Set<string>): number {
  if (a.size === 0 || b.size === 0) {
    return 0;
  }
  const [small, large] = a.size <= b.size ? [a, b] : [b, a];
  let shared = 0;
  for (const token of small) {
    if (large.has(token)) {
      shared++;
    }
  }
  return (2 * shared) / (a.size + b.size);
}

/**
 * Parameter-free Intent-Schema-Overlap score in [0, 1].
 *
 * Combines intent-vs-name overlap (weighted higher — the paper weights
 * name/intent match most heavily) with intent-vs-(description + schema)
 * overlap.
 */
export function intentSchemaOverlapScore(intent: string, entry: ToolManifestEntry): number {
  const intentTokens = new Set<string>(tokenize(intent));
  if (intentTokens.size === 0) {
    return 0;
  }

  const nameTokens = new Set<string>(tokenizeIdentifier(entry.name));
  const bodyTokens = new Set<string>([...tokenize(entry.description), ...entry.schemaTokens]);

  const nameOverlap = diceOverlap(intentTokens, nameTokens);
  const bodyOverlap = diceOverlap(intentTokens, bodyTokens);

  return Math.min(1, nameOverlap * 0.6 + bodyOverlap * 0.4);
}

/**
 * Rank tools by Intent-Schema-Overlap and return the gated-on set for a turn.
 *
 * Tools are sorted by score descending. `alwaysInclude` names bypass both the
 * `minScore` threshold and the `topK` budget; everything else must clear
 * `minScore` and fit within `topK`.
 */
export function rankToolsForIntent(
  intent: string,
  entries: ToolManifestEntry[],
  options: SelectToolsOptions = {},
): SelectedTool[] {
  const topK = options.topK ?? 10;
  const minScore = options.minScore ?? 0.05;
  const alwaysInclude = new Set<string>(options.alwaysInclude ?? []);

  const scored = entries
    .map((entry) => ({ name: entry.name, score: intentSchemaOverlapScore(intent, entry) }))
    .sort((a, b) => b.score - a.score);

  const result: SelectedTool[] = [];
  let budgetUsed = 0;
  for (const candidate of scored) {
    if (alwaysInclude.has(candidate.name)) {
      result.push(candidate);
    } else if (candidate.score >= minScore && budgetUsed < topK) {
      result.push(candidate);
      budgetUsed++;
    }
  }
  return result;
}
