import { Tool } from '../core/tool';
import { ToolkitManager } from '../core/tools/platform-api-tools/manage-tools-tool';
import { ToolAttentionOptions, ToolAttentionResult, ToolRelevanceScorer } from './tool-relevance';

/**
 * Interface representing an MCP server tool registration handle
 */
interface MCPToolHandle {
  enable(): void;
  disable(): void;
}

/**
 * Interface for dynamic tool control
 */
interface DynamicTool {
  instance: Tool<any, any>;
  mcpTool: MCPToolHandle; // Reference to the MCP server tool
  enabled: boolean;
  enabledByDefault: boolean; // Track the original default state
}

/**
 * Manages dynamic tool registration, enabling, and disabling
 */
export class DynamicToolManager implements ToolkitManager {
  private readonly dynamicTools: Map<string, DynamicTool> = new Map();

  private readonly relevanceScorer: ToolRelevanceScorer = new ToolRelevanceScorer();

  /**
   * Register a tool for dynamic management
   */
  registerTool(tool: Tool<any, any>, mcpTool: MCPToolHandle): void {
    // Store the tool reference for dynamic control
    const enabledByDefault = tool.enabledByDefault ?? true; // Default to true if not specified
    const initialEnabled = enabledByDefault;

    this.dynamicTools.set(tool.name, {
      instance: tool,
      mcpTool: mcpTool,
      enabled: initialEnabled,
      enabledByDefault: enabledByDefault,
    });

    // If the tool should be disabled by default, disable it after registration
    if (!enabledByDefault) {
      mcpTool.disable();
    }
  }

  /**
   * Enable a specific tool
   */
  enableTool(toolName: string): boolean {
    const dynamicTool = this.dynamicTools.get(toolName);
    if (!dynamicTool) {
      return false;
    }

    if (!dynamicTool.enabled) {
      dynamicTool.mcpTool.enable();
      dynamicTool.enabled = true;
    }
    return true;
  }

  /**
   * Disable a specific tool
   */
  disableTool(toolName: string): boolean {
    const dynamicTool = this.dynamicTools.get(toolName);
    if (!dynamicTool) {
      return false;
    }

    if (dynamicTool.enabled) {
      dynamicTool.mcpTool.disable();
      dynamicTool.enabled = false;
    }
    return true;
  }

  /**
   * Check if a tool is currently enabled
   */
  isToolEnabled(toolName: string): boolean {
    const dynamicTool = this.dynamicTools.get(toolName);
    return dynamicTool ? dynamicTool.enabled : false;
  }

  /**
   * Check if a tool is enabled by default
   */
  isToolEnabledByDefault(toolName: string): boolean {
    const dynamicTool = this.dynamicTools.get(toolName);
    return dynamicTool ? dynamicTool.enabledByDefault : true;
  }

  /**
   * Get list of all available tools and their status
   */
  getToolsStatus(): Record<string, boolean> {
    const status: Record<string, boolean> = {};
    this.dynamicTools.forEach((dynamicTool, toolName) => {
      status[toolName] = dynamicTool.enabled;
    });
    return status;
  }

  /**
   * Get list of all dynamic tool names
   */
  getDynamicToolNames(): string[] {
    return Array.from(this.dynamicTools.keys());
  }

  /**
   * Get list of all available tools with their current and default status
   */
  getDetailedToolsStatus(): Record<string, { enabled: boolean; enabledByDefault: boolean }> {
    const status: Record<string, { enabled: boolean; enabledByDefault: boolean }> = {};
    this.dynamicTools.forEach((dynamicTool, toolName) => {
      status[toolName] = {
        enabled: dynamicTool.enabled,
        enabledByDefault: dynamicTool.enabledByDefault,
      };
    });
    return status;
  }

  /**
   * Reset a tool to its default enabled state
   */
  resetToolToDefault(toolName: string): boolean {
    const dynamicTool = this.dynamicTools.get(toolName);
    if (!dynamicTool) {
      return false;
    }

    if (dynamicTool.enabledByDefault && !dynamicTool.enabled) {
      dynamicTool.mcpTool.enable();
      dynamicTool.enabled = true;
      return true;
    } else if (!dynamicTool.enabledByDefault && dynamicTool.enabled) {
      dynamicTool.mcpTool.disable();
      dynamicTool.enabled = false;
      return true;
    }

    return true;
  }

  /**
   * Get all registered dynamic tools (for internal use)
   */
  getAllDynamicTools(): Map<string, DynamicTool> {
    return this.dynamicTools;
  }

  /**
   * Apply per-query "Tool Attention" gating: score every registered tool
   * against the query, enable the relevant subset, and disable the rest to
   * shrink the per-turn tool-schema payload (the MCP tax). Off-domain queries
   * (nothing relevant) leave the catalog at its static defaults so the agent is
   * never gated out of every tool; the agent-driven manage_tools path remains a
   * fallback regardless.
   */
  applyToolAttention(query: string, options: ToolAttentionOptions = {}): ToolAttentionResult {
    const { minScore = 0, maxEnabled, preserveToolNames = [] } = options;
    const preserve = new Set(preserveToolNames);

    const tools = Array.from(this.dynamicTools.values()).map((dynamicTool) => dynamicTool.instance);
    const scored = this.relevanceScorer.scoreTools(query, tools);
    const scores: Record<string, number> = {};
    for (const { name, score } of scored) scores[name] = score;

    const relevant = scored.filter((entry) => entry.score > minScore);
    if (relevant.length === 0) {
      return { query, enabled: [], disabled: [], scores, fallback: true };
    }

    const enabledByAttention = new Set<string>();
    for (const { name } of relevant) {
      enabledByAttention.add(name);
      if (maxEnabled !== undefined && enabledByAttention.size >= maxEnabled) break;
    }

    const enabled: string[] = [];
    const disabled: string[] = [];
    this.dynamicTools.forEach((dynamicTool, name) => {
      if (preserve.has(name)) {
        this.enableTool(name);
        enabled.push(name);
      } else if (enabledByAttention.has(name)) {
        this.enableTool(name);
        enabled.push(name);
      } else {
        this.disableTool(name);
        disabled.push(name);
      }
    });

    return { query, enabled, disabled, scores, fallback: false };
  }

  /**
   * Clear all registered tools (for cleanup)
   */
  clear(): void {
    this.dynamicTools.clear();
  }
}
