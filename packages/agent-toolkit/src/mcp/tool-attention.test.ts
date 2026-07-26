import { z } from 'zod';
import { Tool, ToolType } from '../core/tool';
import { ToolAnnotations } from '@modelcontextprotocol/sdk/types';
import {
  buildToolManifest,
  intentSchemaOverlapScore,
  rankToolsForIntent,
  tokenize,
  tokenizeIdentifier,
} from './tool-attention';

// Minimal Tool factory for manifest/scoring tests (no API client needed).
function makeTool(name: string, description: string, schema: Record<string, any> = {}): Tool<any, any> {
  return {
    name,
    type: ToolType.READ,
    annotations: { audience: [] } as ToolAnnotations,
    getDescription: () => description,
    getInputSchema: () => schema as any,
    execute: async () => ({ content: '' }),
  };
}

const searchItemsSchema = {
  query: z.string().describe('Free-text search query'),
  board_id: z.number().describe('Board identifier').optional(),
};

const createDocSchema = {
  title: z.string().describe('Document title'),
  content: z.string().describe('Document body content'),
};

const actionSchema = {
  action: z.enum(['enable', 'disable', 'status', 'reset']).describe('Action to perform on a tool'),
};

describe('tool-attention tokenizer', () => {
  it('lowercases, strips punctuation and stopwords, dedupes', () => {
    expect(tokenize('Search for the SEARCH items, please!')).toEqual(['search', 'items']);
  });

  it('returns [] for empty / stopword-only input', () => {
    expect(tokenize('')).toEqual([]);
    expect(tokenize('the a an of to')).toEqual([]);
  });

  it('splits snake_case / camelCase / kebab identifiers without stopword filtering', () => {
    expect(tokenizeIdentifier('list_boards')).toEqual(['list', 'boards']);
    expect(tokenizeIdentifier('createDoc')).toEqual(['create', 'doc']);
    expect(tokenizeIdentifier('manage-tools')).toEqual(['manage', 'tools']);
  });
});

describe('tool-attention manifest', () => {
  it('extracts name + description + param names + param descriptions + enum values', () => {
    const entry = buildToolManifest(makeTool('search_items', 'Search items and boards', searchItemsSchema));
    expect(entry.name).toBe('search_items');
    expect(entry.description).toBe('Search items and boards');
    // Name + description tokens
    expect(entry.schemaTokens.has('search')).toBe(true);
    expect(entry.schemaTokens.has('items')).toBe(true);
    expect(entry.schemaTokens.has('boards')).toBe(true);
    // Param name tokens
    expect(entry.schemaTokens.has('query')).toBe(true);
    expect(entry.schemaTokens.has('board')).toBe(true);
    expect(entry.schemaTokens.has('identifier')).toBe(true);
  });

  it('includes enum option tokens in the manifest', () => {
    const entry = buildToolManifest(makeTool('manage_tools', 'Manage available tools', actionSchema));
    expect(entry.schemaTokens.has('enable')).toBe(true);
    expect(entry.schemaTokens.has('disable')).toBe(true);
    expect(entry.schemaTokens.has('action')).toBe(true);
  });
});

describe('tool-attention overlap scoring', () => {
  it('scores an intent-relevant tool above an irrelevant one', () => {
    const intent = 'search for items about the marketing board';
    const searchEntry = buildToolManifest(makeTool('search_items', 'Search for items and boards', searchItemsSchema));
    const docEntry = buildToolManifest(makeTool('create_doc', 'Create a new document in a workspace', createDocSchema));

    const searchScore = intentSchemaOverlapScore(intent, searchEntry);
    const docScore = intentSchemaOverlapScore(intent, docEntry);

    expect(searchScore).toBeGreaterThan(docScore);
    expect(searchScore).toBeGreaterThan(0);
    expect(docScore).toBe(0);
  });

  it('returns 0 when the intent has no scorable tokens', () => {
    const entry = buildToolManifest(makeTool('search_items', 'Search items', searchItemsSchema));
    expect(intentSchemaOverlapScore('the a of to', entry)).toBe(0);
  });
});

describe('tool-attention ranking', () => {
  const entries = [
    buildToolManifest(makeTool('search_items', 'Search for items and boards', searchItemsSchema)),
    buildToolManifest(makeTool('create_doc', 'Create a new document in a workspace', createDocSchema)),
    buildToolManifest(makeTool('list_boards', 'List all boards in the account')),
  ];

  it('gates ON only intent-relevant tools and respects topK', () => {
    const selected = rankToolsForIntent('search for items about the marketing board', entries, { topK: 5 });
    const names = selected.map((s) => s.name);
    expect(names).toContain('search_items');
    expect(names).not.toContain('create_doc');
  });

  it('forces alwaysInclude tools on regardless of score or budget', () => {
    const selected = rankToolsForIntent('hello world nothing matches', entries, {
      topK: 0,
      alwaysInclude: ['create_doc'],
    });
    const names = selected.map((s) => s.name);
    expect(names).toContain('create_doc');
    expect(names).not.toContain('search_items');
  });

  it('applies the minScore threshold', () => {
    const selected = rankToolsForIntent('search for items', entries, { minScore: 0.99 });
    expect(selected).toHaveLength(0);
  });
});
