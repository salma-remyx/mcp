import { Tool } from '../core/tool';

/**
 * Parameter-free "Tool Attention" relevance scorer.
 *
 * Adapted from "Tool Attention Is All You Need: Dynamic Tool Gating and Lazy
 * Schema Loading for Eliminating the MCP/Tools Tax in Scalable Agentic
 * Workflows" (arXiv:2604.21816). The paper introduces a per-query relevance
 * signal ("Tool Attention") used to gate which tool schemas are injected into
 * a turn, shrinking the 10k-60k-token "MCP tax" of eager, stateless schema
 * injection. Its learned attention head is substituted here with a
 * parameter-free lexical proxy: a tool's relevance to a query is the fraction
 * of the query's content tokens that appear in the tool's own surface text —
 * its name, description, and input-schema field labels/descriptions (i.e. the
 * very text the MCP schema payload would otherwise inject every turn). The
 * gating itself is applied by `DynamicToolManager.applyToolAttention`, which
 * leaves the agent-driven `manage_tools` path intact as a fallback.
 */

// A small, dependency-free English stop list so scoring keys on content words
// rather than connectives ("search the items for me" -> search, items).
const STOPWORDS = new Set<string>([
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
  'me',
  'my',
  'of',
  'on',
  'or',
  'that',
  'the',
  'their',
  'this',
  'to',
  'us',
  'was',
  'we',
  'were',
  'will',
  'with',
  'you',
  'your',
  'about',
  'all',
  'any',
  'can',
  'do',
  'get',
  'how',
  'into',
  'no',
  'not',
  'out',
  'so',
  'some',
  'there',
  'up',
  'use',
  'what',
  'when',
  'which',
  'who',
  'why',
]);

export interface ToolScore {
  name: string;
  /** Relevance score in [0, 1]. */
  score: number;
}

export interface ToolAttentionOptions {
  /**
   * Minimum relevance score in [0, 1] for a tool to be kept enabled. A tool is
   * considered relevant when its score is strictly greater than this value, so
   * the default of `0` means "shares at least one content token with the query".
   */
  minScore?: number;
  /**
   * Cap on how many tools are enabled (highest-scoring first). Tools above the
   * cap are disabled even if relevant. Defaults to no cap.
   */
  maxEnabled?: number;
  /**
   * Tool names that must never be disabled by attention gating, regardless of
   * score — e.g. the `manage_tools` fallback so the agent can always recover.
   */
  preserveToolNames?: string[];
}

export interface ToolAttentionResult {
  query: string;
  /** Tools that were left enabled by this pass. */
  enabled: string[];
  /** Tools that were gated off (disabled) by this pass. */
  disabled: string[];
  /** Per-tool relevance scores, keyed by tool name. */
  scores: Record<string, number>;
  /**
   * True when nothing in the catalog was relevant to the query. In that case
   * the catalog is left at its static defaults rather than gating the agent
   * out of every tool; the manual `manage_tools` path remains available.
   */
  fallback: boolean;
}

function tokenize(text: string): string[] {
  if (!text) return [];
  return text
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((token) => token.length > 1 && !STOPWORDS.has(token));
}

export class ToolRelevanceScorer {
  /**
   * Build the searchable surface text for a tool: name + description + the
   * labels and descriptions of its input-schema fields. This is exactly the
   * text the MCP schema payload would inject into a turn. All introspection is
   * best-effort so a malformed tool can never break scoring.
   */
  private buildToolDocument(tool: Tool<any, any>): string {
    const parts: string[] = [tool.name];

    let description = '';
    try {
      if (typeof tool.getDescription === 'function') description = tool.getDescription();
    } catch {
      description = '';
    }
    if (description) parts.push(description);

    let schema: unknown = undefined;
    try {
      schema = typeof tool.getInputSchema === 'function' ? tool.getInputSchema() : undefined;
    } catch {
      schema = undefined;
    }
    if (schema && typeof schema === 'object') {
      for (const [key, field] of Object.entries(schema as Record<string, unknown>)) {
        parts.push(key);
        const fieldDescription = (field as { description?: unknown } | null)?.description;
        if (typeof fieldDescription === 'string') parts.push(fieldDescription);
      }
    }

    return parts.join(' ');
  }

  /**
   * Score a single tool against a query in [0, 1]: the fraction of the query's
   * content tokens covered by the tool's surface text. Returns 0 for an empty
   * query or a tool with no surface text.
   */
  scoreTool(query: string, tool: Tool<any, any>): number {
    const queryTokens = tokenize(query);
    if (queryTokens.length === 0) return 0;
    const documentTokens = new Set(tokenize(this.buildToolDocument(tool)));
    if (documentTokens.size === 0) return 0;
    let matched = 0;
    for (const token of queryTokens) {
      if (documentTokens.has(token)) matched += 1;
    }
    return matched / queryTokens.length;
  }

  /**
   * Score every tool against the query and return them most-relevant first.
   */
  scoreTools(query: string, tools: Array<Tool<any, any>>): ToolScore[] {
    return tools
      .map((tool) => ({ name: tool.name, score: this.scoreTool(query, tool) }))
      .sort((a, b) => b.score - a.score || (a.name < b.name ? -1 : 1));
  }
}
