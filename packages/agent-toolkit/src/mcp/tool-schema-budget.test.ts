import { MondayAgentToolkit } from './toolkit';
import { ToolSchemaBudget, estimateSchemaTokenCost } from './tool-schema-budget';
import { Tool, ToolType } from '../core/tool';
import { ToolAnnotations } from '@modelcontextprotocol/sdk/types';
import { getFilteredToolInstances } from '../utils/tools/tools-filtering.utils';
import { z } from 'zod';

jest.mock('../utils/tools/tools-filtering.utils', () => ({
  getFilteredToolInstances: jest.fn(),
}));

const mockGetFilteredToolInstances = getFilteredToolInstances as jest.MockedFunction<typeof getFilteredToolInstances>;

function makeTool(name: string, schema: Record<string, any>): Tool<any, any> {
  return {
    name,
    type: ToolType.READ,
    annotations: { audience: [] } as ToolAnnotations,
    enabledByDefault: true,
    getDescription: () => `Description for ${name}`,
    getInputSchema: () => schema,
    execute: jest.fn(),
  } as unknown as Tool<any, any>;
}

describe('ToolSchemaBudget (MCP Tax meter)', () => {
  describe('estimateSchemaTokenCost', () => {
    it('returns 0 for empty or undefined schemas', () => {
      expect(estimateSchemaTokenCost(undefined)).toBe(0);
      expect(estimateSchemaTokenCost(null)).toBe(0);
      expect(estimateSchemaTokenCost('')).toBe(0);
    });

    it('approximates tokens as chars / 4', () => {
      const schema = { a: 1, b: 'two' };
      const expected = Math.ceil(JSON.stringify(schema).length / 4);
      expect(estimateSchemaTokenCost(schema)).toBe(expected);
    });

    it('handles a pre-serialized string', () => {
      const serialized = '{"a":1}';
      expect(estimateSchemaTokenCost(serialized)).toBe(Math.ceil(serialized.length / 4));
    });
  });

  describe('budget module', () => {
    it('aggregates per-tool token cost and exposes lazy descriptors', () => {
      const budget = new ToolSchemaBudget();
      budget.registerTool(makeTool('search_items', { query: z.string(), limit: z.number().optional() }));
      budget.registerTool(makeTool('list_boards', {}));

      const entries = budget.getBudget();
      expect(entries.map((e) => e.name)).toEqual(['search_items', 'list_boards']);
      expect(entries[0].paramNames).toEqual(['query', 'limit']);
      expect(entries[0].estimatedTokens).toBeGreaterThan(0);
      expect(entries[1].estimatedTokens).toBe(0); // empty schema

      expect(budget.getTotalEstimatedTokens()).toBe(entries[0].estimatedTokens + entries[1].estimatedTokens);

      const descriptors = budget.getLazyDescriptors();
      expect(descriptors).toHaveLength(2);
      expect(Object.keys(descriptors[0]).sort()).toEqual(['description', 'name', 'paramNames']);
    });

    it('falls back to the raw shape when a schema cannot be serialized', () => {
      const budget = new ToolSchemaBudget();
      // A non-Zod shape: zodToJsonSchema(z.object(...)) throws, so the meter
      // falls back to estimating cost from the raw object instead of crashing.
      budget.registerTool(makeTool('odd_tool', { bogus: { not: 'a zod schema' } }));

      const entry = budget.getBudget()[0];
      expect(entry.estimatedTokens).toBeGreaterThan(0);
      expect(entry.paramNames).toEqual(['bogus']);
    });
  });

  describe('integration with MondayAgentToolkit', () => {
    beforeEach(() => {
      jest.clearAllMocks();
    });

    it('builds the budget only when enableToolSchemaBudget is set', () => {
      mockGetFilteredToolInstances.mockReturnValue([makeTool('search_items', { query: z.string() })] as any);

      // Flag off: no budget is built.
      const off = new MondayAgentToolkit({ mondayApiToken: 'test-token' });
      expect(off.getToolSchemaBudget()).toBeUndefined();

      // Flag on: budget is populated with the registered tool's tax.
      const on = new MondayAgentToolkit({
        mondayApiToken: 'test-token',
        toolsConfiguration: { enableToolSchemaBudget: true },
      });
      const budget = on.getToolSchemaBudget();
      expect(budget).toBeInstanceOf(ToolSchemaBudget);
      expect(budget!.getTotalEstimatedTokens()).toBeGreaterThan(0);
      expect(budget!.getBudget()[0].name).toBe('search_items');
      expect(budget!.getLazyDescriptors()[0].paramNames).toEqual(['query']);
    });
  });
});
