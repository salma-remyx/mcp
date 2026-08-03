import { Tool, ToolType } from '../../core/tool';
import { ToolAnnotations } from '@modelcontextprotocol/sdk/types';
import { tokenize, scoreToolRelevance, gateTools } from '../tools/tool-gating.utils';

interface MockTool extends Tool<any, any> {
  name: string;
  type: ToolType;
  annotations: ToolAnnotations;
  getDescription: jest.Mock;
  getInputSchema: jest.Mock;
  execute: jest.Mock;
}

function makeTool(name: string, description: string): MockTool {
  return {
    name,
    type: ToolType.READ,
    annotations: { audience: [] },
    getDescription: jest.fn().mockReturnValue(description),
    getInputSchema: jest.fn().mockReturnValue({}),
    execute: jest.fn(),
  };
}

describe('tokenize', () => {
  it('lower-cases and splits on non-alphanumeric runs', () => {
    expect(tokenize('Create_Doc NOW!')).toEqual(['create', 'doc', 'now']);
  });

  it('drops stop-words and tokens shorter than two characters', () => {
    expect(tokenize('I want to list the boards')).toEqual(['want', 'list', 'boards']);
  });

  it('returns an empty array for empty input', () => {
    expect(tokenize('')).toEqual([]);
  });
});

describe('scoreToolRelevance', () => {
  it('returns 0 when the query shares no tokens with the tool', () => {
    const tool = makeTool('list_boards', 'List all boards in the account');
    expect(scoreToolRelevance(tool, 'create a new document')).toBe(0);
  });

  it('returns 0 for an empty query', () => {
    const tool = makeTool('create_doc', 'Create a new document');
    expect(scoreToolRelevance(tool, '')).toBe(0);
  });

  it('scores a name match higher than an equal description-only match', () => {
    const nameMatch = makeTool('create_doc', 'Unrelated description text');
    const descMatch = makeTool('unrelated_tool', 'Create a new document here');

    expect(scoreToolRelevance(nameMatch, 'create doc')).toBeGreaterThan(scoreToolRelevance(descMatch, 'create doc'));
  });

  it('scores 1 when every query token is covered by the name', () => {
    const tool = makeTool('create_doc', 'Whatever');
    expect(scoreToolRelevance(tool, 'create doc')).toBe(1);
  });
});

describe('gateTools', () => {
  const createDoc = makeTool('create_doc', 'Create a new document in a workspace');
  const listBoards = makeTool('list_boards', 'List all boards in the account');
  const searchItems = makeTool('search_items', 'Search for items across boards');
  const allTools = [createDoc, listBoards, searchItems];

  it('returns all tools unchanged when no query is provided', () => {
    expect(gateTools(allTools, undefined)).toBe(allTools);
    expect(gateTools(allTools, '   ')).toBe(allTools);
  });

  it('returns only the query-relevant tool(s)', () => {
    const gated = gateTools(allTools, 'create a new document');
    expect(gated).toHaveLength(1);
    expect(gated[0].name).toBe('create_doc');
  });

  it('falls back to all tools when nothing matches', () => {
    expect(gateTools(allTools, 'weather forecast')).toBe(allTools);
  });

  it('always keeps force-included tools even when they do not match the query', () => {
    const gated = gateTools(allTools, 'create a new document', { alwaysInclude: ['search_items'] });
    const names = gated.map((t) => t.name);
    expect(names).toContain('create_doc');
    expect(names).toContain('search_items');
    expect(names).not.toContain('list_boards');
  });

  it('respects maxTools, ranking matches by relevance', () => {
    const gated = gateTools(allTools, 'create document boards', { maxTools: 1 });
    expect(gated).toHaveLength(1);
    // 'create'/'document' hit create_doc by name AND description, outranking list_boards (name only on 'boards').
    expect(gated[0].name).toBe('create_doc');
  });
});
