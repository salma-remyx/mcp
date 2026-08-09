import { DynamicToolManager } from './dynamic-tool-manager';
import { scoreToolsForQuery, selectRelevantTools, tokenize } from './tool-gate';
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

function makeTool(name: string, description: string): MockTool {
  return {
    name,
    type: ToolType.READ,
    annotations: { audience: [] },
    enabledByDefault: true,
    getDescription: jest.fn().mockReturnValue(description),
    getInputSchema: jest.fn().mockReturnValue({}),
    execute: jest.fn(),
  };
}

function makeHandle(): MockMCPToolHandle {
  return { enable: jest.fn(), disable: jest.fn() };
}

describe('tool-gate (query-driven relevance gate)', () => {
  describe('tokenize', () => {
    it('splits snake_case names and drops stopwords', () => {
      expect(tokenize('create_board')).toEqual(['create', 'board']);
      expect(tokenize('Please create a new board!')).toEqual(['create', 'new', 'board']);
    });
  });

  describe('scoreToolsForQuery', () => {
    const descriptors = [
      { name: 'create_board', description: 'Create a new board with columns and groups' },
      { name: 'list_items', description: 'Read items and rows from a board' },
      { name: 'search_users', description: 'Find people and accounts by email' },
    ];

    it('ranks the matching tool above the unrelated tool', () => {
      const scored = scoreToolsForQuery('create a new board', descriptors);
      const byName = Object.fromEntries(scored.map((s) => [s.name, s.score]));
      expect(byName['create_board']).toBeGreaterThan(byName['search_users']);
      expect(byName['create_board']).toBeGreaterThan(0);
      expect(byName['search_users']).toBe(0);
    });

    it('returns zero scores for an opaque query', () => {
      const scored = scoreToolsForQuery('xyzzy qwerty', descriptors);
      expect(scored.every((s) => s.score === 0)).toBe(true);
    });
  });

  describe('selectRelevantTools', () => {
    const descriptors = [
      { name: 'create_board', description: 'Create a new board with columns and groups' },
      { name: 'list_items', description: 'Read items and rows from a board' },
      { name: 'search_users', description: 'Find people and accounts by email' },
    ];

    it('keeps the relevant tool and prunes the rest', () => {
      const result = selectRelevantTools('create a new board', descriptors, {
        topK: 1,
        minScore: 0.03,
        alwaysOn: [],
      });
      expect(result.enabled).toContain('create_board');
      expect(result.disabled).toContain('search_users');
    });

    it('keeps always-on tools regardless of score', () => {
      const result = selectRelevantTools('create a new board', descriptors, {
        topK: 1,
        minScore: 0.03,
        alwaysOn: ['search_users'],
      });
      expect(result.enabled).toContain('search_users');
    });

    it('does not prune anything when there is no signal', () => {
      const result = selectRelevantTools('xyzzy qwerty', descriptors, { topK: 1, minScore: 0.03 });
      expect(result.disabled).toEqual([]);
      expect(result.enabled).toHaveLength(descriptors.length);
    });
  });

  describe('DynamicToolManager.applyQueryGate (integration)', () => {
    let manager: DynamicToolManager;

    beforeEach(() => {
      manager = new DynamicToolManager();
      // Register an irrelevant discovery tool plus three capability tools.
      manager.registerTool(makeTool('manage_tools', 'Discover and manage available tools'), makeHandle());
      manager.registerTool(makeTool('create_board', 'Create a new board with columns and groups'), makeHandle());
      manager.registerTool(makeTool('list_items', 'Read items and rows from a board'), makeHandle());
      manager.registerTool(makeTool('search_users', 'Find people and accounts by email'), makeHandle());
    });

    it('enables the relevant tool and disables the unrelated one', () => {
      const result = manager.applyQueryGate('create a new board', { topK: 1, minScore: 0.03 });

      expect(result.enabled).toContain('create_board');
      expect(manager.isToolEnabled('create_board')).toBe(true);
      expect(result.disabled).toContain('search_users');
      expect(manager.isToolEnabled('search_users')).toBe(false);
    });

    it('keeps the discovery tool enabled by default so the agent can recover', () => {
      manager.applyQueryGate('create a new board', { topK: 1, minScore: 0.03 });
      expect(manager.isToolEnabled('manage_tools')).toBe(true);
    });

    it('leaves the full set active for an opaque query', () => {
      const result = manager.applyQueryGate('xyzzy qwerty', { topK: 1, minScore: 0.03 });
      expect(result.disabled).toEqual([]);
      expect(result.enabled).toHaveLength(4);
    });
  });
});
