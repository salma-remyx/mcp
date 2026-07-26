import { MondayAgentToolkit } from './toolkit';
import { gateTools, getToolVocabulary, scoreToolRelevance, tokenize, GatableTool } from './tool-gating';
import { ToolType } from '../core/tool';
import { getFilteredToolInstances } from '../utils/tools/tools-filtering.utils';
import { z } from 'zod';

// Mock the ApiClient so the toolkit can construct without network deps.
jest.mock('@mondaydotcomorg/api', () => ({
  ApiClient: jest.fn().mockImplementation(() => ({})),
}));

// Mock the tool factory so we control exactly which tools are registered.
jest.mock('../utils/tools/tools-filtering.utils', () => ({
  getFilteredToolInstances: jest.fn(),
}));

const mockGetFilteredToolInstances = getFilteredToolInstances as jest.MockedFunction<typeof getFilteredToolInstances>;

/** Build a minimal gatable tool for the pure unit tests below. */
function makeTool(name: string, description: string, schema?: Record<string, unknown>): GatableTool {
  return {
    name,
    getDescription: () => description,
    getInputSchema: () => schema,
  };
}

describe('tool-gating — tokenizer', () => {
  it('splits camelCase, kebab, snake and path separators', () => {
    expect(tokenize('searchItems')).toEqual(['search', 'items']);
    expect(tokenize('create-doc')).toEqual(['create', 'doc']);
    expect(tokenize('list_boards/v2')).toEqual(['list', 'boards', 'v2']);
  });

  it('drops stopwords and single characters, case-insensitively', () => {
    expect(tokenize('Search for the items!')).toEqual(['search', 'items']);
    expect(tokenize('')).toEqual([]);
  });
});

describe('tool-gating — vocabulary', () => {
  it('puts name, description and argument tokens into separate buckets', () => {
    const tool = makeTool('create-doc', 'Create a new document in a workspace', {
      boardId: z.string().describe('the board identifier'),
    });
    const vocab = getToolVocabulary(tool);
    expect(vocab.name.has('create')).toBe(true);
    expect(vocab.name.has('doc')).toBe(true);
    expect(vocab.description.has('document')).toBe(true);
    expect(vocab.args.has('board')).toBe(true); // arg name + its description share "board"
    expect(vocab.args.has('identifier')).toBe(true);
  });

  it('handles tools with no schema', () => {
    const vocab = getToolVocabulary(makeTool('ping', 'health check'));
    expect(vocab.args.size).toBe(0);
    expect(vocab.name.has('ping')).toBe(true);
  });
});

describe('tool-gating — scoreToolRelevance', () => {
  it('scores 1.0 when every query token is a name token', () => {
    const tool = makeTool('search-items', 'Search for items and boards');
    expect(scoreToolRelevance('search items', tool)).toBe(1);
  });

  it('scores between 0 and 1 when matches are description-level only', () => {
    const tool = makeTool('create-doc', 'Create a new document in a workspace');
    const score = scoreToolRelevance('document', tool);
    expect(score).toBeGreaterThan(0);
    expect(score).toBeLessThan(1);
  });

  it('scores 0 when there is no overlap', () => {
    const tool = makeTool('list-boards', 'List all boards in a workspace');
    expect(scoreToolRelevance('document', tool)).toBe(0);
  });

  it('scores 0 for an empty query', () => {
    expect(scoreToolRelevance('', makeTool('any-tool', 'does things'))).toBe(0);
  });
});

describe('tool-gating — gateTools', () => {
  const tools: GatableTool[] = [
    makeTool('search-items', 'Search for items and boards across the workspace'),
    makeTool('create-doc', 'Create a new document in a workspace'),
    makeTool('list-boards', 'List all boards in a workspace'),
  ];

  it('keeps only tools with overlap and ranks by score descending', () => {
    const gated = gateTools('search items', tools);
    expect(gated.map((g) => g.name)).toEqual(['search-items']);
    expect(gated[0].score).toBe(1);
  });

  it('respects topK', () => {
    const gated = gateTools('boards', tools, { topK: 1 });
    // "boards" matches list-boards (name) and search-items (description); top-1 wins
    expect(gated).toHaveLength(1);
    expect(gated[0].name).toBe('list-boards');
  });

  it('applies a minScore floor', () => {
    const gated = gateTools('boards', tools, { minScore: 1 });
    // only a full name match (list-boards) clears a 1.0 floor
    expect(gated.map((g) => g.name)).toEqual(['list-boards']);
  });

  it('breaks score ties by name for deterministic order', () => {
    const tied: GatableTool[] = [makeTool('zeta-tool', 'shared thing'), makeTool('alpha-tool', 'shared thing')];
    const gated = gateTools('shared', tied);
    expect(gated.map((g) => g.name)).toEqual(['alpha-tool', 'zeta-tool']);
  });

  it('returns nothing for a query that matches no tool', () => {
    expect(gateTools('totally unrelated gibberish', tools)).toHaveLength(0);
  });
});

describe('tool-gating — integration with MondayAgentToolkit', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  function buildToolkit(
    tools: Array<{ name: string; description: string; schema?: Record<string, unknown> }>,
  ): MondayAgentToolkit {
    mockGetFilteredToolInstances.mockReturnValue(
      tools.map(
        (t) =>
          ({
            name: t.name,
            type: ToolType.READ,
            annotations: { audience: [] },
            enabledByDefault: true,
            getDescription: jest.fn().mockReturnValue(t.description),
            getInputSchema: jest.fn().mockReturnValue(t.schema ?? {}),
            execute: jest.fn().mockResolvedValue({ content: 'ok' }),
          }) as any,
      ),
    );
    return new MondayAgentToolkit({ mondayApiToken: 'test-token' });
  }

  it('gates the active surface through the existing DynamicToolManager', () => {
    const toolkit = buildToolkit([
      { name: 'search-items', description: 'Search for items and boards across the workspace' },
      { name: 'create-doc', description: 'Create a new document in a workspace' },
      { name: 'list-boards', description: 'List all boards in a workspace' },
    ]);

    // All three start enabled.
    expect(toolkit.isToolEnabled('search-items')).toBe(true);
    expect(toolkit.isToolEnabled('create-doc')).toBe(true);
    expect(toolkit.isToolEnabled('list-boards')).toBe(true);

    const result = toolkit.gateToolsForQuery('search items', { topK: 1 });

    // The relevant tool stays enabled; the rest are disabled.
    expect(result.kept).toEqual(['search-items']);
    expect(result.disabled.sort()).toEqual(['create-doc', 'list-boards']);
    expect(toolkit.isToolEnabled('search-items')).toBe(true);
    expect(toolkit.isToolEnabled('create-doc')).toBe(false);
    expect(toolkit.isToolEnabled('list-boards')).toBe(false);
    expect(result.scores['search-items']).toBe(1);
  });

  it('restores a wider surface for a different query on a fresh toolkit', () => {
    const toolkit = buildToolkit([
      { name: 'search-items', description: 'Search for items and boards across the workspace' },
      { name: 'create-doc', description: 'Create a new document in a workspace' },
    ]);

    const docResult = toolkit.gateToolsForQuery('create document');
    expect(docResult.kept).toEqual(['create-doc']);
    expect(toolkit.isToolEnabled('search-items')).toBe(false);
    expect(toolkit.isToolEnabled('create-doc')).toBe(true);
  });

  it('disables every tool when the query shares no tokens', () => {
    const toolkit = buildToolkit([{ name: 'search-items', description: 'Search for items' }]);
    const result = toolkit.gateToolsForQuery('totally unrelated gibberish');
    expect(result.kept).toHaveLength(0);
    expect(result.disabled).toEqual(['search-items']);
    expect(toolkit.isToolEnabled('search-items')).toBe(false);
  });
});
