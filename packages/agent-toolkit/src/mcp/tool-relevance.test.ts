import { DynamicToolManager } from './dynamic-tool-manager';
import { ToolRelevanceScorer } from './tool-relevance';
import { Tool, ToolType } from '../core/tool';
import { ToolAnnotations } from '@modelcontextprotocol/sdk/types';

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

function makeTool(name: string, description: string, opts: { enabledByDefault?: boolean } = {}): MockTool {
  return {
    name,
    type: ToolType.READ,
    annotations: { audience: [] },
    enabledByDefault: opts.enabledByDefault,
    getDescription: jest.fn().mockReturnValue(description),
    getInputSchema: jest.fn().mockReturnValue({}),
    execute: jest.fn(),
  };
}

function makeHandle(): MockMCPToolHandle {
  return { enable: jest.fn(), disable: jest.fn() };
}

describe('ToolRelevanceScorer', () => {
  const scorer = new ToolRelevanceScorer();

  it('scores 0 for an empty or stopword-only query', () => {
    const tool = makeTool('search_items', 'Search for items and boards');
    expect(scorer.scoreTool('', tool)).toBe(0);
    expect(scorer.scoreTool('the and is', tool)).toBe(0);
  });

  it('scores 1.0 when every query content token appears in the tool surface text', () => {
    const tool = makeTool('search_items', 'Search for items and boards across the workspace');
    expect(scorer.scoreTool('search items', tool)).toBe(1);
  });

  it('scores partial coverage as a fraction of query tokens', () => {
    const tool = makeTool('search_items', 'Search for items and boards');
    // query tokens: search, items, invoices -> 2 of 3 covered
    expect(scorer.scoreTool('search items invoices', tool)).toBeCloseTo(2 / 3, 5);
  });

  it('folds input-schema field labels and descriptions into the document', () => {
    const tool = makeTool('create_doc', 'Create a document');
    tool.getInputSchema = jest.fn().mockReturnValue({
      title: { description: 'The document title' },
      board_id: { description: 'Target board identifier' },
    });
    // "board" is covered via the board_id field description
    expect(scorer.scoreTool('board', tool)).toBe(1);
  });

  it('returns tools sorted most-relevant first', () => {
    const search = makeTool('search_items', 'Search for items');
    const docs = makeTool('read_docs', 'Read document content');
    const ranked = scorer.scoreTools('search items', [docs, search]);
    expect(ranked[0].name).toBe('search_items');
    expect(ranked[1].name).toBe('read_docs');
  });
});

describe('DynamicToolManager.applyToolAttention (integration)', () => {
  let manager: DynamicToolManager;
  let searchHandle: MockMCPToolHandle;
  let docsHandle: MockMCPToolHandle;
  let manageHandle: MockMCPToolHandle;

  beforeEach(() => {
    manager = new DynamicToolManager();
    searchHandle = makeHandle();
    docsHandle = makeHandle();
    manageHandle = makeHandle();
    manager.registerTool(
      makeTool('search_items', 'Search for items and boards across the monday.com workspace by query terms'),
      searchHandle,
    );
    manager.registerTool(makeTool('read_docs', 'Read and retrieve content from monday.com documents'), docsHandle);
    manager.registerTool(
      makeTool('manage_tools', 'Discover and manage available monday.com tools and their state'),
      manageHandle,
    );
  });

  it('enables query-relevant tools and gates off the rest', () => {
    const result = manager.applyToolAttention('search for items about invoices');

    expect(result.fallback).toBe(false);
    expect(result.enabled).toContain('search_items');
    expect(result.disabled).toContain('read_docs');
    expect(manager.isToolEnabled('search_items')).toBe(true);
    expect(manager.isToolEnabled('read_docs')).toBe(false);
    expect(docsHandle.disable).toHaveBeenCalled();
  });

  it('falls back to the static defaults when the query is off-domain', () => {
    // All three tools were enabled by default at registration, so an
    // off-domain query must not gate any of them off.
    const result = manager.applyToolAttention('what is the weather today');

    expect(result.fallback).toBe(true);
    expect(result.enabled).toEqual([]);
    expect(result.disabled).toEqual([]);
    expect(manager.isToolEnabled('search_items')).toBe(true);
    expect(manager.isToolEnabled('read_docs')).toBe(true);
    expect(searchHandle.disable).not.toHaveBeenCalled();
  });

  it('preserves tools pinned via preserveToolNames regardless of score', () => {
    const result = manager.applyToolAttention('search for items', {
      preserveToolNames: ['manage_tools'],
    });

    expect(result.enabled).toContain('manage_tools');
    expect(result.disabled).not.toContain('manage_tools');
    expect(manager.isToolEnabled('manage_tools')).toBe(true);
  });

  it('caps the number of enabled tools to the highest-scoring', () => {
    // A second tool that shares a token but scores lower than search_items.
    manager.registerTool(makeTool('search_boards', 'search across boards and columns'), makeHandle());

    const result = manager.applyToolAttention('search items', { maxEnabled: 1 });

    expect(result.enabled).toHaveLength(1);
    expect(result.enabled).toContain('search_items');
    expect(result.disabled).toContain('search_boards');
  });
});
