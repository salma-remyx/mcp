import { MondayAgentToolkit } from './toolkit';
import { ToolType } from '../core/tool';
import { getFilteredToolInstances } from '../utils/tools/tools-filtering.utils';
import { z } from 'zod';
import {
  estimateToolTokens,
  materializeToolSchema,
  scoreToolRelevance,
  selectToolsForTurn,
  tokenize,
} from './tool-gating';

// Same module mocks used by toolkit.test.ts so MondayAgentToolkit can construct without the
// monday.com API client or the real tool registry.
jest.mock('@mondaydotcomorg/api', () => ({
  ApiClient: jest.fn().mockImplementation(() => ({})),
}));

jest.mock('../utils/tools/tools-filtering.utils', () => ({
  getFilteredToolInstances: jest.fn(),
}));

jest.mock('../core/tools/platform-api-tools/manage-tools-tool', () => ({
  ManageToolsTool: jest.fn().mockImplementation(() => ({
    name: 'manage-tools',
    type: ToolType.READ,
    annotations: { audience: [] },
    enabledByDefault: true,
    getDescription: jest.fn().mockReturnValue('Manage tools'),
    getInputSchema: jest.fn().mockReturnValue({}),
    execute: jest.fn(),
    setToolkitManager: jest.fn(),
  })),
}));

const mockGetFilteredToolInstances = getFilteredToolInstances as jest.MockedFunction<typeof getFilteredToolInstances>;

// Minimal Tool-shaped mock spanning a few domains.
const makeTool = (name: string, description: string, schema: Record<string, unknown> = {}): any => ({
  name,
  type: ToolType.READ,
  annotations: { audience: [] },
  enabledByDefault: true,
  getDescription: jest.fn().mockReturnValue(description),
  getInputSchema: jest.fn().mockReturnValue(schema),
  execute: jest.fn().mockResolvedValue({ content: 'ok' }),
});

const sampleTools = [
  makeTool('create_board', 'Create a new board in the workspace', { board_name: z.string() }),
  makeTool('search_items', 'Search for items across boards', { query: z.string() }),
  makeTool('create_doc', 'Create a document', { title: z.string() }),
  makeTool('list_users', 'List all users and teams', {}),
];

describe('tool-gating (Tool Attention / MCP Tax reduction)', () => {
  describe('relevance proxy and tax meter', () => {
    it('tokenize lowercases, splits on non-alphanumerics, and drops glue words', () => {
      expect(tokenize('Create a NEW board!')).toEqual(new Set(['create', 'new', 'board']));
    });

    it('estimateToolTokens grows with schema size and is always positive', () => {
      const small = makeTool('t1', 'tiny', {});
      const big = makeTool('t2', 'big', {
        a: z.string(),
        b: z.number(),
        c: z.string().describe('a longer description here'),
      });
      expect(estimateToolTokens(small)).toBeGreaterThan(0);
      expect(estimateToolTokens(big)).toBeGreaterThan(estimateToolTokens(small));
    });

    it('materializeToolSchema returns undefined for schemaless tools', () => {
      const noSchema = {
        ...makeTool('t', 'd'),
        getInputSchema: jest.fn().mockReturnValue(undefined),
      };
      expect(materializeToolSchema(noSchema as any)).toBeUndefined();
    });

    it('scoreToolRelevance ranks a relevant tool above an irrelevant one', () => {
      const relevant = makeTool('create_board', 'Create a new board in the workspace');
      const irrelevant = makeTool('list_users', 'List all users and teams');
      expect(scoreToolRelevance(relevant as any, 'create a new board')).toBeGreaterThan(
        scoreToolRelevance(irrelevant as any, 'create a new board'),
      );
    });
  });

  describe('selectToolsForTurn', () => {
    it('passes all tools through with zero savings when there is no query', () => {
      const decision = selectToolsForTurn(sampleTools as any, '');
      expect(decision.enabledNames).toHaveLength(4);
      expect(decision.deferredNames).toHaveLength(0);
      expect(decision.tokensSaved).toBe(0);
    });

    it('defers low-relevance tools under a token budget and reports savings', () => {
      const total = sampleTools.reduce((sum, t) => sum + estimateToolTokens(t as any), 0);
      // Budget the single most-relevant tool's cost with a floor of 1 => exactly one tool kept.
      const topCost = estimateToolTokens(sampleTools[0] as any);
      const decision = selectToolsForTurn(sampleTools as any, 'create a new board', {
        tokenBudget: topCost,
        minTools: 1,
      });

      expect(decision.enabledNames).toEqual(['create_board']);
      expect(decision.deferredNames).toContain('list_users');
      expect(decision.totalSchemaTokens).toBe(total);
      expect(decision.gatedSchemaTokens).toBeLessThan(total);
      expect(decision.tokensSaved).toBeGreaterThan(0);
    });
  });

  describe('integration with MondayAgentToolkit (call site)', () => {
    let toolkit: MondayAgentToolkit;
    const byName = Object.fromEntries(sampleTools.map((t) => [t.name, t]));

    beforeEach(() => {
      jest.clearAllMocks();
      mockGetFilteredToolInstances.mockReturnValue(sampleTools as any);
      toolkit = new MondayAgentToolkit({ mondayApiToken: 'test-token' });
    });

    it('exposes all tools eagerly by default', () => {
      expect(toolkit.getDynamicToolNames()).toEqual(
        expect.arrayContaining(['create_board', 'search_items', 'create_doc', 'list_users']),
      );
      expect(toolkit.isToolEnabled('create_board')).toBe(true);
      expect(toolkit.isToolEnabled('list_users')).toBe(true);
    });

    it('gateToolsForQuery with no budget is a no-op (all tools stay enabled)', () => {
      const decision = toolkit.gateToolsForQuery('create a new board');
      expect(decision.tokensSaved).toBe(0);
      expect(toolkit.isToolEnabled('create_board')).toBe(true);
      expect(toolkit.isToolEnabled('list_users')).toBe(true);
    });

    it('gates tools for a query and drives the enable/disable path', () => {
      const topCost = estimateToolTokens(byName.create_board as any);
      const decision = toolkit.gateToolsForQuery('create a new board', topCost, 1);

      // Most-relevant tool exposed; the irrelevant ones deferred via disableTool.
      expect(toolkit.isToolEnabled('create_board')).toBe(true);
      expect(toolkit.isToolEnabled('list_users')).toBe(false);
      expect(toolkit.isToolEnabled('create_doc')).toBe(false);
      expect(decision.tokensSaved).toBeGreaterThan(0);
    });

    it('re-gating with a different query flips which tools are exposed', () => {
      const boardCost = estimateToolTokens(byName.create_board as any);
      toolkit.gateToolsForQuery('create a new board', boardCost, 1);
      expect(toolkit.isToolEnabled('create_board')).toBe(true);
      expect(toolkit.isToolEnabled('list_users')).toBe(false);

      const usersCost = estimateToolTokens(byName.list_users as any);
      toolkit.gateToolsForQuery('show me all users and teams', usersCost, 1);
      expect(toolkit.isToolEnabled('list_users')).toBe(true);
      expect(toolkit.isToolEnabled('create_board')).toBe(false);
    });
  });
});
