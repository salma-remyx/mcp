import {
  estimateTokens,
  gateTools,
  getToolStub,
  scoreTool,
  tokenize,
  ToolStub,
} from './tool-attention';
import { MondayAgentToolkit } from './toolkit';
import { ToolType } from '../core/tool';
import { getFilteredToolInstances } from '../utils/tools/tools-filtering.utils';
import { z } from 'zod';

// The toolkit constructor builds a real monday API client and resolves the tool
// set through getFilteredToolInstances; mock both so the integration test can
// drive the wiring without a network or token.
jest.mock('@mondaydotcomorg/api', () => ({
  ApiClient: jest.fn().mockImplementation(() => ({})),
}));
jest.mock('../utils/tools/tools-filtering.utils', () => ({
  getFilteredToolInstances: jest.fn(),
}));

const mockGetFilteredToolInstances = getFilteredToolInstances as jest.MockedFunction<
  typeof getFilteredToolInstances
>;

// Minimal Tool-shaped mocks with realistic monday.com-style surfaces.
function makeTool(name: string, description: string, shape: Record<string, any> = {}) {
  return {
    name,
    type: ToolType.READ,
    annotations: { audience: [] },
    enabledByDefault: true,
    getDescription: jest.fn().mockReturnValue(description),
    getInputSchema: jest.fn().mockReturnValue(shape),
    execute: jest.fn().mockResolvedValue({ content: 'ok' }),
  };
}

const searchItems = makeTool('search_items', 'Search for items and boards across the workspace', {
  query: z.string().describe('The search query text'),
  limit: z.number().optional().describe('Maximum number of results to return'),
});
const createDoc = makeTool('create_doc', 'Create a new document inside a workspace', {
  workspace_id: z.string().describe('Target workspace identifier'),
  name: z.string().describe('Document title'),
});
const userContext = makeTool('get_user_context', "Get the current user's boards, teams, and environment", {});

describe('tool-attention (unit)', () => {
  describe('tokenize', () => {
    it('lowercases, splits on non-alphanumeric, drops stopwords and short tokens', () => {
      expect(tokenize('Search the Items!')).toEqual(['search', 'items']);
      expect(tokenize('')).toEqual([]);
      expect(tokenize('a the an of')).toEqual([]);
    });
  });

  describe('estimateTokens', () => {
    it('returns 0 for empty input and ~4 chars/token otherwise', () => {
      expect(estimateTokens('')).toBe(0);
      expect(estimateTokens('abcdefgh')).toBe(2);
    });
  });

  describe('getToolStub', () => {
    it('derives a compact stub without materializing the full schema', () => {
      const stub: ToolStub = getToolStub(searchItems);
      expect(stub.name).toBe('search_items');
      expect(stub.paramCount).toBe(2);
      expect(stub.paramNames).toEqual(['query', 'limit']);
      expect(stub.estimatedSchemaTokens).toBeGreaterThan(0);
    });

    it('handles tools with no input schema', () => {
      const stub = getToolStub(userContext);
      expect(stub.paramCount).toBe(0);
      expect(stub.paramNames).toEqual([]);
    });
  });

  describe('scoreTool', () => {
    it('scores full query coverage as 1.0', () => {
      const stub = getToolStub(searchItems);
      expect(scoreTool(['search', 'items'], stub)).toBe(1);
    });

    it('scores zero when no query term hits the tool surface', () => {
      const stub = getToolStub(createDoc);
      expect(scoreTool(['search', 'items'], stub)).toBe(0);
    });

    it('returns 0 for an empty query', () => {
      expect(scoreTool([], getToolStub(searchItems))).toBe(0);
    });
  });

  describe('gateTools', () => {
    it('keeps relevant tools and defers the rest, reporting token savings', () => {
      const result = gateTools([searchItems, createDoc, userContext], 'search items');

      expect(result.gated.map((g) => g.stub.name)).toEqual(['search_items']);
      expect(result.gated[0].score).toBe(1);
      expect(result.deferred.map((g) => g.stub.name).sort()).toEqual(['create_doc', 'get_user_context']);
      expect(result.tokensSaved).toBeGreaterThan(0);
      expect(result.tokensSaved).toBeLessThan(result.totalSchemaTokens);
    });

    it('keeps every tool when the query carries no signal', () => {
      const result = gateTools([searchItems, createDoc], 'the a');
      expect(result.gated).toHaveLength(2);
      expect(result.deferred).toHaveLength(0);
      expect(result.tokensSaved).toBe(0);
    });

    it('respects an explicit minScore threshold', () => {
      // search_items has score 1.0, create_doc has 0.0 against this query.
      const result = gateTools([searchItems, createDoc], 'search', { minScore: 1 });
      expect(result.gated.map((g) => g.stub.name)).toEqual(['search_items']);
    });

    it('caps the kept set to topK', () => {
      const result = gateTools([searchItems, createDoc, userContext], '', { topK: 1 });
      expect(result.gated).toHaveLength(1);
      expect(result.deferred).toHaveLength(2);
      expect(result.tokensSaved).toBeGreaterThan(0);
    });
  });
});

describe('tool-attention (toolkit wiring)', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockGetFilteredToolInstances.mockReturnValue([searchItems, createDoc, userContext]);
  });

  it('getToolStubs returns a lazy stub for every registered tool', () => {
    const toolkit = new MondayAgentToolkit({ mondayApiToken: 'test-token' });
    const stubs = toolkit.getToolStubs();

    expect(stubs).toHaveLength(3);
    expect(stubs.map((s) => s.name).sort()).toEqual(['create_doc', 'get_user_context', 'search_items']);
    expect(stubs.every((s) => s.estimatedSchemaTokens >= 0)).toBe(true);
  });

  it('getToolTokenReport measures the per-turn schema tax', () => {
    const toolkit = new MondayAgentToolkit({ mondayApiToken: 'test-token' });
    const report = toolkit.getToolTokenReport();

    expect(Object.keys(report.perTool).sort()).toEqual(['create_doc', 'get_user_context', 'search_items']);
    const sum = Object.values(report.perTool).reduce((a, b) => a + b, 0);
    expect(report.totalSchemaTokens).toBe(sum);
    expect(report.totalSchemaTokens).toBeGreaterThan(0);
  });

  it('gateTools gates the toolkit tool set against a query', () => {
    const toolkit = new MondayAgentToolkit({ mondayApiToken: 'test-token' });
    const result = toolkit.gateTools('search items');

    expect(result.gated.map((g) => g.stub.name)).toEqual(['search_items']);
    expect(result.deferred.map((g) => g.stub.name).sort()).toEqual(['create_doc', 'get_user_context']);
    expect(result.tokensSaved).toBeGreaterThan(0);
  });
});
