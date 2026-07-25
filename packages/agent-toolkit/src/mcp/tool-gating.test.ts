import { DynamicToolManager } from './dynamic-tool-manager';
import { Tool, ToolType } from '../core/tool';
import { ToolAnnotations } from '@modelcontextprotocol/sdk/types';
import {
  estimateTokenTax,
  gate,
  scoreToolIntent,
  selectToolsByIntent,
  summarizeDescription,
  type ToolGateMeta,
} from './tool-gating';

// Mock types for testing (mirror tests/dynamic-tool-manager.test.ts)
interface MockMCPToolHandle {
  enable: jest.Mock;
  disable: jest.Mock;
}

interface MockTool extends Tool<any, any> {
  name: string;
  type: ToolType;
  annotations: ToolAnnotations;
  enabledByDefault?: boolean;
  getDescription: jest.Mock;
  getInputSchema: jest.Mock;
  execute: jest.Mock;
}

const DOCS_TOOL: ToolGateMeta = {
  name: 'create_doc',
  description: 'Create a new document inside a monday.com workspace.',
  paramNames: ['workspace_id', 'document_name', 'content'],
};
const SEARCH_TOOL: ToolGateMeta = {
  name: 'search_items',
  description: 'Search for items and boards across the workspace.',
  paramNames: ['query', 'limit'],
};
const FORECAST_TOOL: ToolGateMeta = {
  name: 'forecast_revenue',
  description: 'Forecast quarterly revenue from CRM deals.',
  paramNames: ['deal_ids', 'quarter'],
};
const METAS: ToolGateMeta[] = [DOCS_TOOL, SEARCH_TOOL, FORECAST_TOOL];

describe('tool-gating (Intent–Schema overlap + lazy schema loader)', () => {
  describe('scoreToolIntent', () => {
    it('scores a schema-overlapping intent higher than an unrelated one', () => {
      const docScore = scoreToolIntent('create a document for the workspace', DOCS_TOOL);
      const forecastScore = scoreToolIntent('create a document for the workspace', FORECAST_TOOL);
      expect(docScore).toBeGreaterThan(0);
      expect(docScore).toBeGreaterThan(forecastScore);
    });

    it('returns 0 for an intent with no overlapping tokens', () => {
      expect(scoreToolIntent('zzz qqq xxx', DOCS_TOOL)).toBe(0);
    });

    it('is bounded to [0,1]', () => {
      const score = scoreToolIntent('document workspace document content', DOCS_TOOL);
      expect(score).toBeGreaterThanOrEqual(0);
      expect(score).toBeLessThanOrEqual(1);
    });
  });

  describe('summarizeDescription', () => {
    it('truncates long descriptions to a single short line', () => {
      const long = 'This is a very long description. It has a second sentence that should be dropped.';
      const out = summarizeDescription(long, 30);
      expect(out.length).toBeLessThanOrEqual(30);
      expect(out.endsWith('…')).toBe(true);
    });

    it('keeps short single-line descriptions intact', () => {
      expect(summarizeDescription('Short one.', 90)).toBe('Short one.');
    });
  });

  describe('gate (two-phase lazy loader)', () => {
    it('emits a summary pool for ALL tools and a smaller gated detail pool', () => {
      const result = gate(METAS, 'forecast revenue', { topK: 1 });
      expect(result.summaryPool).toHaveLength(3);
      expect(result.detailPool).toEqual(['forecast_revenue']);
      expect(result.scores[0].name).toBe('forecast_revenue');
    });

    it('keeps always-on tools in the detail pool even with a zero score', () => {
      const result = gate(METAS, 'qqq zzz', { topK: 1, alwaysOn: ['search_items'] });
      expect(result.detailPool).toContain('search_items');
    });

    it('filters low-relevance tools out of the detail pool via minScore', () => {
      const result = gate(METAS, 'document', { topK: 5, minScore: 0.9 });
      expect(result.detailPool).toContain('create_doc');
      expect(result.detailPool).not.toContain('search_items');
      expect(result.detailPool).not.toContain('forecast_revenue');
    });

    it('reports token-tax savings (gated pool injects fewer tokens than all tools)', () => {
      const { tokenTax } = gate(METAS, 'forecast revenue', { topK: 1 });
      expect(tokenTax.allTools).toBeGreaterThan(tokenTax.gated);
      expect(tokenTax.saved).toBe(tokenTax.allTools - tokenTax.gated);
    });
  });

  describe('selectToolsByIntent', () => {
    it('returns just the detail-pool names', () => {
      expect(selectToolsByIntent(METAS, 'search items', { topK: 1 })).toEqual(['search_items']);
    });
  });

  describe('estimateTokenTax', () => {
    it('computes allTools / gated / saved consistently', () => {
      const tax = estimateTokenTax(METAS, ['forecast_revenue']);
      expect(tax.gated).toBeLessThan(tax.allTools);
      expect(tax.saved).toBe(tax.allTools - tax.gated);
    });
  });

  // Integration: imports the existing DynamicToolManager and exercises the
  // gateByIntent wiring added in dynamic-tool-manager.ts.
  describe('DynamicToolManager.gateByIntent (integration)', () => {
    const makeHandle = (): MockMCPToolHandle => ({ enable: jest.fn(), disable: jest.fn() });

    const makeTool = (name: string, description: string, schema: Record<string, unknown>): MockTool => ({
      name,
      type: ToolType.READ,
      annotations: { audience: [] },
      enabledByDefault: true,
      getDescription: jest.fn().mockReturnValue(description),
      getInputSchema: jest.fn().mockReturnValue(schema),
      execute: jest.fn(),
    });

    it('enables intent-relevant tools and disables the rest via existing enable/disable', () => {
      const manager = new DynamicToolManager();
      const docHandle = makeHandle();
      const searchHandle = makeHandle();
      const forecastHandle = makeHandle();
      manager.registerTool(
        makeTool('create_doc', 'Create a new document inside a workspace.', { workspace_id: {}, document_name: {} }),
        docHandle,
      );
      manager.registerTool(
        makeTool('search_items', 'Search for items and boards across the workspace.', { query: {} }),
        searchHandle,
      );
      manager.registerTool(
        makeTool('forecast_revenue', 'Forecast quarterly revenue from CRM deals.', { deal_ids: {} }),
        forecastHandle,
      );

      const enabled = manager.gateByIntent('search for items', { topK: 1 });

      expect(enabled).toEqual(['search_items']);
      expect(manager.isToolEnabled('search_items')).toBe(true);
      expect(manager.isToolEnabled('create_doc')).toBe(false);
      expect(manager.isToolEnabled('forecast_revenue')).toBe(false);
      // search was already enabled at registration (no extra enable call);
      // the off-topic tools get disabled through the existing primitive.
      expect(searchHandle.enable).not.toHaveBeenCalled();
      expect(docHandle.disable).toHaveBeenCalledTimes(1);
      expect(forecastHandle.disable).toHaveBeenCalledTimes(1);
    });

    it('respects alwaysOn when gating', () => {
      const manager = new DynamicToolManager();
      const docHandle = makeHandle();
      const searchHandle = makeHandle();
      manager.registerTool(makeTool('create_doc', 'Create a new document inside a workspace.', {}), docHandle);
      manager.registerTool(
        makeTool('search_items', 'Search for items and boards across the workspace.', {}),
        searchHandle,
      );

      const enabled = manager.gateByIntent('search for items', { topK: 1, alwaysOn: ['create_doc'] });

      expect(enabled).toEqual(expect.arrayContaining(['search_items', 'create_doc']));
      expect(manager.isToolEnabled('create_doc')).toBe(true);
      expect(docHandle.disable).not.toHaveBeenCalled();
    });
  });
});
