/**
 * Lazy schema loading for MCP tool listings.
 *
 * By default every registered tool eager-injects its full input schema into the
 * `tools/list` payload handed to the LLM. In multi-tool deployments this
 * "tools tax" can dominate the context window. This module implements an opt-in
 * lazy mode following the core mechanism of "Tool Attention Is All You Need:
 * Dynamic Tool Gating and Lazy Schema Loading for Eliminating the MCP/Tools
 * Tax in Scalable Agentic Workflows" (arXiv:2604.21816): emit a compact summary
 * of every tool's schema and hydrate the full schema only for a small
 * "promoted" top-k subset.
 *
 * Adapted port (Mode 2): the paper's learned tool-attention scorer is replaced
 * by a parameter-free token-overlap proxy, and the paper's separate middleware /
 * evaluation framework is cut — this hooks the toolkit's existing
 * getToolsForMcp / getSchemaForTool path instead. The compact-summary-pool +
 * top-k-promote mechanism is preserved at full fidelity.
 */

/**
 * Structural JSON-Schema keys retained in a compact summary. Everything else
 * (descriptions, examples, defaults, bounds, $schema, titles) is verbose
 * human-facing metadata that inflates the payload without changing which
 * arguments a tool accepts.
 */
const STRUCTURAL_KEYS: ReadonlySet<string> = new Set([
  'type',
  'enum',
  'items',
  'properties',
  'required',
  'anyOf',
  'oneOf',
  'allOf',
  '$ref',
  'additionalProperties',
]);

/**
 * Reduce a full JSON Schema to a compact summary skeleton: keep only the
 * structural shape (types, required flags, nested items/properties) and drop
 * verbose human-facing metadata. Recurses into every retained node.
 */
export function compactJsonSchema(schema: any): any {
  if (schema === null || typeof schema !== 'object') {
    return schema;
  }

  if (Array.isArray(schema)) {
    return schema.map(compactJsonSchema);
  }

  const compacted: Record<string, any> = {};
  for (const key of Object.keys(schema)) {
    if (!STRUCTURAL_KEYS.has(key)) {
      continue;
    }
    if (key === 'properties') {
      // `properties` is a map of {paramName: schema}, not a schema itself, so
      // keep every parameter name and recurse into each parameter's schema.
      const props = schema[key] as Record<string, any>;
      const compactedProps: Record<string, any> = {};
      for (const propName of Object.keys(props)) {
        compactedProps[propName] = compactJsonSchema(props[propName]);
      }
      compacted[key] = compactedProps;
    } else {
      compacted[key] = compactJsonSchema(schema[key]);
    }
  }
  return compacted;
}

export interface ToolSummary {
  name: string;
  description: string;
}

export interface LazySchemaOptions {
  /** Relevance signal used to pick which tools keep their full schema. */
  query?: string;
  /** Number of tools to promote to a full schema. Defaults to 3. */
  topK?: number;
  /** Explicit tool names to always promote, overriding top-k selection. */
  promote?: string[];
}

const DEFAULT_TOP_K = 3;

const STOPWORDS: ReadonlySet<string> = new Set([
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
  'these',
  'to',
  'with',
  'tool',
  'tools',
]);

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((token) => token.length > 1 && !STOPWORDS.has(token));
}

/**
 * Parameter-free relevance proxy (stands in for the paper's learned
 * tool-attention): weighted token overlap between a query and a tool's name +
 * description. Name tokens count double to bias selection toward
 * purpose-relevant tools.
 */
export function scoreTool(tool: ToolSummary, queryTokens: string[]): number {
  if (queryTokens.length === 0) {
    return 0;
  }

  const nameTokens = new Set(tokenize(tool.name));
  const descriptionTokens = new Set(tokenize(tool.description));
  let score = 0;
  for (const token of queryTokens) {
    if (nameTokens.has(token)) {
      score += 2;
    }
    if (descriptionTokens.has(token)) {
      score += 1;
    }
  }
  return score;
}

/**
 * Select the set of tool names that should keep their full schema under lazy
 * mode. Explicit `promote` names are always included (on-demand hydration);
 * the remaining budget up to `topK` is filled by the most query-relevant tools,
 * with ties broken by name for deterministic output. When no query is supplied,
 * the first `topK` tools (by name) fill the budget.
 */
export function selectPromotedTools(tools: ToolSummary[], options: LazySchemaOptions = {}): Set<string> {
  const promoted = new Set<string>(options.promote ?? []);
  const topK = options.topK ?? DEFAULT_TOP_K;

  if (promoted.size >= topK || tools.length <= promoted.size) {
    return promoted;
  }

  const remaining = topK - promoted.size;
  const queryTokens = tokenize(options.query ?? '');

  const ranked = tools
    .filter((tool) => !promoted.has(tool.name))
    .map((tool) => ({ name: tool.name, score: scoreTool(tool, queryTokens) }))
    .sort((a, b) => b.score - a.score || a.name.localeCompare(b.name));

  for (const { name } of ranked.slice(0, remaining)) {
    promoted.add(name);
  }
  return promoted;
}

/**
 * Resolve the schema to emit for a tool under lazy mode: the full schema when
 * the tool is in the promoted set, otherwise its compact summary.
 */
export function resolveLazySchema(fullJsonSchema: any, toolName: string, promoted: Set<string>): any {
  return promoted.has(toolName) ? fullJsonSchema : compactJsonSchema(fullJsonSchema);
}
