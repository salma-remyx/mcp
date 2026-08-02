import { Tool } from '../../core/tool';

/**
 * Lazy tool-schema loading — a token-aware reduction of the per-turn "tools tax".
 *
 * Adapted from: "Tool Attention Is All You Need: Dynamic Tool Gating and Lazy
 * Schema Loading for Eliminating the MCP/Tools Tax in Scalable Agentic
 * Workflows" (arXiv:2604.21816). The paper keeps a compact *summary pool* for
 * every tool and promotes the full JSON schema only for the top-k tools gated
 * as relevant to the current turn, instead of eagerly injecting every tool's
 * full schema on every turn.
 *
 * Mode 2 (adapted port): the paper's *ISO intent router* (a learned/LLM intent
 * classifier) is replaced here by a parameter-free token-overlap relevance
 * proxy — `scoreToolRelevance` — that needs no model and no extra round trip.
 * The paper's *state-aware gate* becomes a deterministic top-k selector. The
 * paper's separate benchmark/eval suite is intentionally out of scope.
 */

/** Default number of tools that keep their full JSON schema when lazy loading is on. */
export const DEFAULT_MAX_FULL_SCHEMAS = 10;

/** JSON-Schema keys that carry prose / enumeration bulk and are dropped from a summary. */
const SUMMARY_DROP_KEYS = new Set([
  'description',
  'enum',
  '$schema',
  'additionalProperties',
  'default',
  'examples',
  '$comment',
  'title',
]);

/** Common English stopwords excluded from relevance matching. */
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
  'at',
  'from',
  'as',
  'it',
  'this',
  'that',
  'my',
  'our',
  'new',
  'i',
  'want',
  'need',
  'please',
  'using',
  'use',
]);

/**
 * Lowercases and splits free text into a deduplicated set of relevance tokens.
 * Separator characters (`_`, `-`, `.`, `/`) are collapsed so `create_column`
 * and `createColumn` tokenize identically.
 */
export const tokenizeText = (text: unknown): string[] => {
  const tokens = String(text ?? '')
    .toLowerCase()
    .replace(/[_\-./]+/g, ' ')
    .split(/[^a-z0-9]+/g)
    .filter((token) => token.length > 1 && !STOPWORDS.has(token));
  return Array.from(new Set(tokens));
};

/**
 * Produces a token-light summary of a JSON Schema by recursively stripping
 * prose (`description`), enumerations (`enum`), and bookkeeping keys
 * (`$schema`, `additionalProperties`, ...). Structure — `type`, `properties`,
 * `required` — is preserved, so the model still understands each tool's shape
 * and can request the full schema (e.g. via the existing manage-tools gate).
 */
export const summarizeJsonSchema = (node: unknown): any => {
  if (Array.isArray(node)) {
    return node.map((entry) => summarizeJsonSchema(entry));
  }
  if (node === null || typeof node !== 'object') {
    return node;
  }

  const summarized: Record<string, any> = {};
  for (const [key, value] of Object.entries(node as Record<string, any>)) {
    if (SUMMARY_DROP_KEYS.has(key)) {
      continue;
    }
    summarized[key] = summarizeJsonSchema(value);
  }
  return summarized;
};

/**
 * Parameter-free relevance proxy (stands in for the paper's learned ISO intent
 * router). Scores a tool against a user query by counting query tokens that
 * also appear in the tool's name, description, and argument names. Higher is
 * more relevant; 0 means no lexical overlap.
 */
export const scoreToolRelevance = (tool: Tool<any, any>, query: string): number => {
  const queryTokens = tokenizeText(query);
  if (queryTokens.length === 0) {
    return 0;
  }

  let rawSchema: Record<string, any> | undefined;
  try {
    rawSchema = tool.getInputSchema?.() as Record<string, any> | undefined;
  } catch {
    rawSchema = undefined;
  }
  const argNames = rawSchema && typeof rawSchema === 'object' ? Object.keys(rawSchema) : [];

  const toolText = [tool.name, tool.getDescription?.() ?? '', ...argNames].join(' ');
  const toolTokens = new Set(tokenizeText(toolText));

  let score = 0;
  for (const token of queryTokens) {
    if (toolTokens.has(token)) {
      score += 1;
    }
  }
  return score;
};

/**
 * State-aware top-k gate. Returns the set of tool names that should keep their
 * full JSON schema for the current turn. Every tool with a positive relevance
 * score is promoted (never dropped for an irrelevant one while slots remain);
 * the set is then filled up to `maxFullSchemas` in original registration order
 * and never exceeds that cap. All other tools fall back to summary schemas.
 */
export const selectRelevantToolNames = (
  tools: Tool<any, any>[],
  query: string,
  maxFullSchemas: number = DEFAULT_MAX_FULL_SCHEMAS,
): Set<string> => {
  const cap = Math.max(0, maxFullSchemas);
  const scored = tools.map((tool, index) => ({
    name: tool.name,
    score: scoreToolRelevance(tool, query),
    index,
  }));

  const positive = scored.filter((entry) => entry.score > 0);
  if (positive.length >= cap) {
    return new Set(positive.slice(0, cap).map((entry) => entry.name));
  }

  const chosenNames = new Set(positive.map((entry) => entry.name));
  const fillers = scored
    .filter((entry) => !chosenNames.has(entry.name))
    .slice(0, cap - positive.length)
    .map((entry) => entry.name);
  for (const name of fillers) {
    chosenNames.add(name);
  }
  return chosenNames;
};
