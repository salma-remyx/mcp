import { Tool } from '../core/tool';

/**
 * Tool Attention — dynamic tool gating and lazy schema loading to reduce the
 * "MCP/Tools Tax": the per-turn token cost of eagerly injecting every tool's
 * full input schema into an agent's context window.
 *
 * Adapted from "Tool Attention Is All You Need: Dynamic Tool Gating and Lazy
 * Schema Loading for Eliminating the MCP/Tools Tax in Scalable Agentic
 * Workflows" (arXiv:2604.21816). This is an ADAPTED PORT (Mode 2):
 *   - The paper's learned "ISO gate" / tool-attention relevance head is replaced
 *     by a PARAMETER-FREE lexical relevance proxy: overlap between the query
 *     tokens and the tool's name + description + parameter names + parameter
 *     descriptions. No model, no training, no weights.
 *   - The paper's two-phase lazy-schema protocol (emit a compact stub first,
 *     materialize the full schema only for gated-in tools) is delivered as a
 *     TARGET-NATIVE capability over the toolkit's existing Tool objects: it
 *     emits compact schema stubs, scores each tool against a query, and reports
 *     the per-turn token savings from deferring the gated-out schemas.
 *
 * It complements — and does not replace — DynamicToolManager's coarse binary
 * enable/disable gating, and it never mutates server state: it is a measurement
 * and recommendation surface the toolkit exposes to callers.
 */

/** Chars-per-token heuristic used to size schema payloads (~4 chars/token). */
const CHARS_PER_TOKEN = 4;

/** Common terms removed before scoring relevance. */
const STOPWORDS = new Set([
  'the', 'a', 'an', 'and', 'or', 'of', 'to', 'in', 'on', 'for', 'with', 'i',
  'is', 'are', 'be', 'my', 'me', 'please', 'want', 'needs', 'need', 'do',
]);

/** Compact, schema-deferred description of a tool (the lazy "stub" phase). */
export interface ToolStub {
  name: string;
  description: string;
  paramCount: number;
  paramNames: string[];
  /** Estimated per-turn tokens the full input schema costs when eagerly injected. */
  estimatedSchemaTokens: number;
}

/** A tool alongside its stub and its relevance to the gating query. */
export interface GatedTool {
  tool: Tool<any, any>;
  stub: ToolStub;
  /** Relevance score in [0, 1]; 0 means no query-term overlap. */
  score: number;
}

/** Result of gating a tool set against a query. */
export interface ToolAttentionResult {
  /** Tools that passed the relevance gate, most relevant first. */
  gated: GatedTool[];
  /** Tools gated out (schemas deferred), in the same sort order. */
  deferred: GatedTool[];
  /** Estimated tokens saved per turn by not injecting the deferred schemas. */
  tokensSaved: number;
  /** Estimated tokens the full (eager) schema set would cost per turn. */
  totalSchemaTokens: number;
}

export interface GateOptions {
  /** Minimum relevance score in [0, 1] to keep a tool. Default: any overlap (score > 0). */
  minScore?: number;
  /** Cap the kept set to the top-K most relevant tools. */
  topK?: number;
}

/** Estimate token count from text length. */
export function estimateTokens(text: string): number {
  if (!text) return 0;
  return Math.ceil(text.length / CHARS_PER_TOKEN);
}

/** Lowercase alphanumeric tokens, length > 1, with stopwords removed. */
export function tokenize(text: string): string[] {
  if (!text) return [];
  return text
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((t) => t.length > 1 && !STOPWORDS.has(t));
}

/** Split a snake_case / kebab-case identifier into lowercase tokens. */
function tokenizeIdentifier(name: string): string[] {
  return tokenize(name.replace(/[_-]+/g, ' '));
}

/** Coarse type label for a Zod schema node, best-effort (e.g. ZodString -> "string"). */
function coarseType(node: any): string {
  const typeName: string | undefined = node?._def?.typeName;
  if (!typeName) return 'value';
  return typeName.replace(/^Zod/, '').toLowerCase() || 'value';
}

/**
 * Build a compact, schema-deferred stub for a tool. The stub exposes the tool's
 * identity and parameter surface WITHOUT materializing the full JSON schema —
 * the lazy-loading payload sent on phase one.
 */
export function getToolStub(tool: Tool<any, any>): ToolStub {
  const description = tool.getDescription() ?? '';
  const shape = tool.getInputSchema() ?? {};
  const entries = Object.entries(shape) as [string, any][];

  const paramNames = entries.map(([name]) => name);
  const serialized = [
    tool.name,
    description,
    ...entries.map(([name, node]) => `${name} ${coarseType(node)} ${node?.description ?? ''}`),
  ].join('\n');

  return {
    name: tool.name,
    description,
    paramCount: paramNames.length,
    paramNames,
    estimatedSchemaTokens: estimateTokens(serialized),
  };
}

/** Token set describing a tool's surface, used for relevance matching. */
function toolTokenSet(stub: ToolStub): Set<string> {
  const tokens = new Set<string>();
  tokenizeIdentifier(stub.name).forEach((t) => tokens.add(t));
  tokenize(stub.description).forEach((t) => tokens.add(t));
  stub.paramNames.forEach((name) => tokenizeIdentifier(name).forEach((t) => tokens.add(t)));
  return tokens;
}

/**
 * Score a tool's relevance to a query as query-token coverage in [0, 1].
 * Parameter-free proxy for the paper's learned ISO / attention gate: every
 * query term that appears in the tool's surface contributes equally.
 */
export function scoreTool(queryTokens: string[], stub: ToolStub): number {
  if (queryTokens.length === 0) return 0;
  const surface = toolTokenSet(stub);
  let hit = 0;
  for (const token of queryTokens) {
    if (surface.has(token)) hit += 1;
  }
  return hit / queryTokens.length;
}

/**
 * Two-phase lazy gate: emit compact stubs for every tool, score each against the
 * query, and split into gated-in (schema would be materialized) vs deferred
 * (schema omitted), reporting the per-turn token savings. Deterministic: ties in
 * score are broken by tool name so callers and tests get a stable ordering.
 */
export function gateTools(
  tools: Tool<any, any>[],
  query: string,
  options?: GateOptions,
): ToolAttentionResult {
  const queryTokens = tokenize(query);
  const hasQuery = queryTokens.length > 0;
  const minScore = options?.minScore;
  const topK = options?.topK;

  const scored: GatedTool[] = tools
    .map((tool) => {
      const stub = getToolStub(tool);
      return { tool, stub, score: scoreTool(queryTokens, stub) };
    })
    .sort((a, b) => {
      if (b.score !== a.score) return b.score - a.score;
      return a.stub.name.localeCompare(b.stub.name);
    });

  const keep = (score: number): boolean => {
    if (!hasQuery) return true; // nothing to gate on -> keep everything
    if (minScore !== undefined) return score >= minScore;
    return score > 0; // default: drop tools with zero query overlap
  };

  const eligible = scored.filter((g) => keep(g.score));
  const gated = topK !== undefined && topK >= 0 ? eligible.slice(0, topK) : eligible;
  const gatedNames = new Set(gated.map((g) => g.stub.name));
  const deferred = scored.filter((g) => !gatedNames.has(g.stub.name));

  const totalSchemaTokens = scored.reduce((sum, g) => sum + g.stub.estimatedSchemaTokens, 0);
  const tokensSaved = deferred.reduce((sum, g) => sum + g.stub.estimatedSchemaTokens, 0);

  return { gated, deferred, tokensSaved, totalSchemaTokens };
}
