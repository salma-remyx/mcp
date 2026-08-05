import { z } from 'zod';
import { zodToJsonSchema } from 'zod-to-json-schema';
import { Tool } from '../core/tool';

/**
 * Tool schema budget — opt-in "MCP Tax" meter and lazy schema descriptors.
 *
 * Adapted from "Tool Attention Is All You Need: Dynamic Tool Gating and Lazy
 * Schema Loading for Eliminating the MCP/Tools Tax in Scalable Agentic
 * Workflows" (arXiv:2604.21816). The paper observes that MCP servers eagerly
 * inject every tool's full JSON schema on every turn, costing an estimated
 * 10k–60k tokens per turn (the "MCP Tax"). The repo's DynamicToolManager +
 * ManageToolsTool already provide the paper's dynamic-gating lever; this
 * module delivers the paper's other lever — lazy schema loading — by
 *
 *   1. measuring the per-turn token cost of each tool's advertised schema
 *      (the MCP Tax), and
 *   2. producing compact "lazy descriptors" (name + description + param
 *      names) that elide the full nested JSON schema, so a caller can defer
 *      loading the verbose schema until a tool is actually selected.
 *
 * Substitutions vs. the paper (Mode 2 adapted port): the paper reports
 * measured token counts from a specific tokenizer; we substitute a
 * parameter-free chars/4 proxy, the standard tokenizer-independent
 * approximation. The paper's full on-demand schema delivery (a second
 * round-trip / companion tool) is intentionally out of scope here — this
 * module ships the meter + descriptor primitive; wiring it into the
 * advertised tool list is a separate, larger behavior change.
 */

// Parameter-free token proxy: ~4 characters per token is the standard
// tokenizer-independent approximation used across the ecosystem.
const CHARS_PER_TOKEN = 4;

/** Per-tool entry in the schema budget. */
export interface ToolSchemaBudgetEntry {
  name: string;
  description: string;
  estimatedTokens: number;
  paramNames: string[];
}

/**
 * Compact lazy descriptor: name, one-line description, and parameter names
 * only — the full nested JSON schema is intentionally elided.
 */
export interface LazyToolDescriptor {
  name: string;
  description: string;
  paramNames: string[];
}

/**
 * Estimate the token cost of a JSON-serializable tool schema using a
 * parameter-free chars/4 proxy. Returns 0 for empty/undefined schemas.
 */
export function estimateSchemaTokenCost(schema: unknown): number {
  if (schema === undefined || schema === null) {
    return 0;
  }
  const serialized = typeof schema === 'string' ? schema : JSON.stringify(schema);
  if (serialized.length === 0) {
    return 0;
  }
  return Math.ceil(serialized.length / CHARS_PER_TOKEN);
}

/**
 * Measures the per-turn "MCP Tax" of a toolkit's tool schemas and produces
 * compact lazy descriptors that elide the full nested schemas.
 */
export class ToolSchemaBudget {
  private readonly entries = new Map<string, ToolSchemaBudgetEntry>();

  /**
   * Register a tool. Its full input schema is serialized once to estimate
   * the token cost; only the compact entry is retained. If the schema
   * cannot be serialized (e.g. a non-Zod shape), the cost falls back to the
   * raw shape so a single misbehaving tool cannot break the meter.
   */
  registerTool(tool: Tool<any, any>): ToolSchemaBudgetEntry {
    const inputSchema = tool.getInputSchema();
    const paramNames = inputSchema ? Object.keys(inputSchema) : [];

    let estimatedTokens = 0;
    if (inputSchema && paramNames.length > 0) {
      try {
        estimatedTokens = estimateSchemaTokenCost(zodToJsonSchema(z.object(inputSchema)));
      } catch {
        estimatedTokens = estimateSchemaTokenCost(inputSchema);
      }
    }

    const entry: ToolSchemaBudgetEntry = {
      name: tool.name,
      description: tool.getDescription(),
      estimatedTokens,
      paramNames,
    };
    this.entries.set(tool.name, entry);
    return entry;
  }

  /** Per-tool budget entries, in registration order. */
  getBudget(): ToolSchemaBudgetEntry[] {
    return Array.from(this.entries.values());
  }

  /** Total estimated per-turn token cost across all registered tools. */
  getTotalEstimatedTokens(): number {
    return this.getBudget().reduce((sum, entry) => sum + entry.estimatedTokens, 0);
  }

  /**
   * Compact lazy descriptors (name + description + param names) that elide
   * the full nested JSON schema. A caller can advertise these instead of the
   * full schemas and load the verbose schema on demand once a tool is
   * selected — the paper's lazy schema loading primitive.
   */
  getLazyDescriptors(): LazyToolDescriptor[] {
    return this.getBudget().map(({ name, description, paramNames }) => ({ name, description, paramNames }));
  }
}
