import { MondayAgentToolkit } from './toolkit';
import { ToolType } from '../core/tool';
import { ApiClient } from '@mondaydotcomorg/api';
import { getFilteredToolInstances } from '../utils/tools/tools-filtering.utils';
import { z } from 'zod';

jest.mock('@mondaydotcomorg/api', () => ({
  ApiClient: jest.fn().mockImplementation(() => ({})),
}));
jest.mock('../utils/tools/tools-filtering.utils', () => ({
  getFilteredToolInstances: jest.fn(),
}));

const mockGetFilteredToolInstances = getFilteredToolInstances as jest.MockedFunction<typeof getFilteredToolInstances>;

// Minimal Tool<any, any>-shaped mocks spanning the repo's actual tool categories.
const docTool = {
  name: 'create_doc',
  type: ToolType.WRITE,
  annotations: { audience: [] },
  enabledByDefault: true,
  getDescription: jest.fn().mockReturnValue('Create a new document in a workspace with blocks'),
  getInputSchema: jest.fn().mockReturnValue({ workspace_id: z.number(), title: z.string() }),
  execute: jest.fn(),
};
const searchTool = {
  name: 'search_items',
  type: ToolType.READ,
  annotations: { audience: [] },
  enabledByDefault: true,
  getDescription: jest.fn().mockReturnValue('Search for items and boards across the workspace'),
  getInputSchema: jest.fn().mockReturnValue({ query: z.string() }),
  execute: jest.fn(),
};
const userTool = {
  name: 'get_user_context',
  type: ToolType.READ,
  annotations: { audience: [] },
  enabledByDefault: true,
  getDescription: jest.fn().mockReturnValue('Get the current users teams and boards'),
  getInputSchema: jest.fn().mockReturnValue({}),
  execute: jest.fn(),
};

describe('MondayAgentToolkit tool gating (Tool Attention)', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockGetFilteredToolInstances.mockReturnValue([docTool, searchTool, userTool]);
  });

  describe('getLazyToolIndex()', () => {
    it('ranks tools relevant to the query first and emits compact entries', () => {
      const toolkit = new MondayAgentToolkit({ mondayApiToken: 'token' });
      const index = toolkit.getLazyToolIndex('create a document for the workspace');

      // Most relevant tool surfaces first.
      expect(index[0].name).toBe('create_doc');
      // Compact phase-1 shape: description + parameter names, no full JSON schema.
      expect(index[0].paramNames).toEqual(expect.arrayContaining(['workspace_id', 'title']));
      expect(index[0]).not.toHaveProperty('schema');
    });
  });

  describe('getRelevantTools()', () => {
    it('returns full MCP descriptors only for the gated subset', () => {
      const toolkit = new MondayAgentToolkit({ mondayApiToken: 'token' });
      const relevant = toolkit.getRelevantTools('search for items');

      const names = relevant.map((tool) => tool.name);
      expect(names).toContain('search_items');
      expect(names).not.toContain('create_doc');
      expect(names).not.toContain('get_user_context');

      // Survivors carry their full schema + handler (phase-2 expansion).
      const survivor = relevant.find((tool) => tool.name === 'search_items');
      expect(survivor).toHaveProperty('schema');
      expect(survivor).toHaveProperty('handler');
      expect(typeof survivor?.handler).toBe('function');
    });

    it('returns every tool when the query carries no intent', () => {
      const toolkit = new MondayAgentToolkit({ mondayApiToken: 'token' });
      const relevant = toolkit.getRelevantTools('   ');
      expect(relevant).toHaveLength(3);
    });
  });

  describe('gateToolsForQuery()', () => {
    it('disables non-relevant tools via the existing dynamic tool manager', () => {
      const toolkit = new MondayAgentToolkit({ mondayApiToken: 'token' });

      // Everything is enabled at rest.
      expect(toolkit.getToolsStatus()).toEqual({
        create_doc: true,
        search_items: true,
        get_user_context: true,
      });

      const result = toolkit.gateToolsForQuery('search for items');
      expect(result.selected).toEqual(['search_items']);
      expect(result.gated).toEqual(expect.arrayContaining(['create_doc', 'get_user_context']));

      const status = toolkit.getToolsStatus();
      expect(status.search_items).toBe(true);
      expect(status.create_doc).toBe(false);
      expect(status.get_user_context).toBe(false);
    });
  });
});
